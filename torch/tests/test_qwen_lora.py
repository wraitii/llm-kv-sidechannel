import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from llmpr_torch.attention import (
    configure_fixed_swa, dense_causal_mask, qwen_mask_mapping,
)
from llmpr_torch.lora import TARGET_MODULES, attach_lora
from llmpr_torch.scored import ScoredRetention, enable_scored_retention
from llmpr_torch.soundness import run_checks
from llmpr_torch.policies import FixedSWA, VariableSWA
from llmpr_torch.tokenization import TokenizedEpisode
from llmpr_torch.training import make_mask


def tiny_config():
    return Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )


def test_tiny_qwen_lora_answer_only_backward():
    torch.manual_seed(7)
    config = tiny_config()
    model = attach_lora(Qwen3ForCausalLM(config), rank=4, alpha=8)
    assert all(any(name.endswith(target) for name in model.targeted_module_names)
               for target in TARGET_MODULES)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[-100, -100, -100, -100, 5, 6]])
    loss = model(input_ids=input_ids, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    assert trainable
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
               for parameter in trainable)
    assert all(parameter.grad is None for parameter in model.parameters()
               if not parameter.requires_grad)


def test_qwen_dense_variable_window_mask_is_batch_invariant():
    torch.manual_seed(11)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11]])
    windows = torch.tensor([2, 4])
    batch_mask = qwen_mask_mapping(dense_causal_mask(
        tokens.shape[1], windows=windows))
    with torch.no_grad():
        batched = model(input_ids=tokens, attention_mask=batch_mask).logits
        singles = []
        for row, window in zip(tokens, windows, strict=True):
            mask = qwen_mask_mapping(dense_causal_mask(
                tokens.shape[1], windows=int(window)))
            singles.append(model(input_ids=row[None], attention_mask=mask).logits)
    torch.testing.assert_close(batched, torch.cat(singles), atol=1e-6, rtol=1e-5)


def test_variable_swa_training_mask_uses_one_window_per_row():
    row = TokenizedEpisode(input_ids=(1, 2, 3, 4, 5, 6),
                           labels=(-100, -100, -100, -100, 5, 6), prompt_length=4,
                           answer_length=2, support_token_spans=())
    actual = make_mask([row, row], VariableSWA(2, 4), "cpu", torch.float32,
                       windows=[2, 4])
    expected = dense_causal_mask(6, windows=torch.tensor([2, 4]))
    torch.testing.assert_close(actual, expected)


def test_training_mask_pins_annotated_memory_positions():
    row = TokenizedEpisode(
        input_ids=tuple(range(8)), labels=(-100,) * 8, prompt_length=8,
        answer_length=0, support_token_spans=(), memory_token_spans=((2, 3),),
    )
    actual = make_mask(
        [row], FixedSWA(2), "cpu", torch.float32,
        retain_memory_tokens=True,
    )[0, 0]
    assert actual[7, 2] == 0
    assert actual[7, 3] == 0
    assert actual[7, 4] < -1e20
    assert actual[1, 2] < -1e20


def test_all_layer_fixed_swa_config_matches_dense_reference_with_eviction():
    torch.manual_seed(13)
    full = Qwen3ForCausalLM(tiny_config()).eval()
    swa_config = configure_fixed_swa(tiny_config(), window=3)
    assert set(swa_config.layer_types) == {"sliding_attention"}
    swa = Qwen3ForCausalLM(swa_config).eval()
    swa.load_state_dict(full.state_dict())
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    mask = qwen_mask_mapping(dense_causal_mask(tokens.shape[1], windows=3))
    with torch.no_grad():
        expected = full(input_ids=tokens, attention_mask=mask).logits
        actual = swa(input_ids=tokens).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_scored_retention_has_exact_budget_and_scorer_gradients():
    torch.manual_seed(17)
    model = enable_scored_retention(
        Qwen3ForCausalLM(tiny_config()), ScoredRetention(3, 2, scoring_delay=0))
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
    loss.backward()
    for layer in model.model.layers:
        scorer = layer.self_attn.retention_scorer
        assert layer.self_attn.last_retention_scores.shape == tokens.shape
        assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
                   for parameter in scorer.parameters())


def test_scored_parameters_can_be_trained_alongside_lora():
    model = enable_scored_retention(
        Qwen3ForCausalLM(tiny_config()), ScoredRetention(3, 2))
    model = attach_lora(model, rank=2, alpha=4)
    for name, parameter in model.named_parameters():
        if ".retention_scorer." in name:
            parameter.requires_grad_(True)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert any("lora_" in name for name in trainable)
    assert any("retention_scorer" in name for name in trainable)


def test_backend_cache_and_no_eviction_restart_soundness():
    torch.manual_seed(19)
    model = Qwen3ForCausalLM(tiny_config())
    report = run_checks(model, torch.tensor([[1, 2, 3, 4, 5, 6]]), window=4)
    assert report["finite"]
    assert report["cached_full_max_abs_error"] < 1e-5
    assert report["no_eviction_restart_max_abs_error"] < 1e-5


def test_soundness_reports_native_swa_eviction_equivalence():
    torch.manual_seed(23)
    full = Qwen3ForCausalLM(tiny_config())
    swa = Qwen3ForCausalLM(configure_fixed_swa(tiny_config(), window=3))
    swa.load_state_dict(full.state_dict())
    report = run_checks(
        full, torch.tensor([[1, 2, 3, 4, 5, 6]]), window=3, swa_model=swa)
    assert report["native_swa_evicted_tokens"] == 3
    assert report["native_swa_dense_max_abs_error"] < 1e-5
