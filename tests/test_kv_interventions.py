import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from llmz.model import PrefixLM
from llmz.inference import DecodeSession, prepare_prefix
from llmz.tokenizer import PairTokenizer, BytesTokenizer
from llmz.transport import RecursiveCarrierPolicy


def small_model(scored=None):
    mx.random.seed(123)
    return PrefixLM(260, 260, 32, 3, 4, 2, 128, dtype=mx.float32,
                    carrier_vocab=2, scored_eviction=scored)


def test_variable_carrier_batches_match_single_row_prefill_and_decode():
    model = small_model()
    tok = PairTokenizer(BytesTokenizer(), BytesTokenizer(), carrier_vocab=2)
    rows = [{"asm": "e2e4 e7e5 g1f3 b8c6", "code": "x"},
            {"asm": "d2d4 d7d5", "code": "y"}]
    policy = RecursiveCarrierPolicy(hidden_tokens=[1, 3], carrier_tokens=[1, 2], group_size=3)
    for transport in (False, True):
        for mode in ("preserve", "restart", "restart-each"):
            batch = DecodeSession(model, *prepare_prefix(tok, rows, 32, policy, transport), window=5, mode=mode)
            singles = [DecodeSession(model, *prepare_prefix(tok, [row], 32, policy, transport),
                                     window=5, mode=mode) for row in rows]
            for _ in range(3):
                for i, single in enumerate(singles):
                    assert mx.allclose(batch.logits[i:i+1], single.logits, atol=2e-6, rtol=2e-5).item()
                batch.advance(mx.array([[260], [260]]))
                for single in singles:
                    single.advance(mx.array([[260]]))


def test_transport_prefill_decode_matches_same_pass_training():
    model = small_model()
    tokens = mx.array([[1, 10, 11, 12, 13, 14, 2, 260, 261]])
    spans = mx.array([[[1, 3, 3], [3, 4, 5]]])
    for window in (None, 4):
        _, cache = model.prefill(tokens[:, :4], mx.ones((1, 4), dtype=mx.bool_), mx.array([7]),
                                 window, spans)
        for t in range(4, tokens.shape[1]):
            logits, cache = model.decode(tokens[:, t:t+1], cache, window)
            direct = model(tokens[:, :t+1], mx.ones((1, t+1), dtype=mx.bool_), mx.array([7]),
                           mx.array([[t]]), spans, window)
            assert mx.allclose(logits, direct, atol=2e-6, rtol=2e-5).item()


def test_restart_removes_old_context_channel_and_is_identity_without_eviction():
    model = small_model()
    tokens = mx.array([[1, 10, 11, 12, 13, 2], [1, 99, 11, 12, 13, 2]])
    valid = mx.ones(tokens.shape, dtype=mx.bool_)
    boundary = mx.array([6, 6])
    original = DecodeSession(model, tokens, valid, boundary, window=3)
    restart = DecodeSession(model, tokens, valid, boundary, window=3, mode="restart")
    assert float(mx.max(mx.abs(original.logits[0] - original.logits[1]))) > 1e-7
    assert mx.allclose(restart.logits[0], restart.logits[1], atol=1e-7).item()
    full = DecodeSession(model, tokens, valid, boundary)
    rebuilt = DecodeSession(model, tokens, valid, boundary, mode="restart-each")
    for _ in range(3):
        assert mx.allclose(full.logits, rebuilt.logits, atol=2e-6, rtol=2e-5).item()
        full.advance(mx.array([[260], [260]]))
        rebuilt.advance(mx.array([[260], [260]]))


def test_scored_hard_budget_causality_and_cached_equivalence():
    model = small_model({"recent_window": 2, "memory_tokens": 2, "score_dim": 8})
    tokens = mx.array([[1, 10, 11, 12, 13, 14, 15, 16, 2]])
    _, cache = model.prefill(tokens[:, :3], mx.ones((1, 3), dtype=mx.bool_), mx.array([9]))
    for t in range(3, tokens.shape[1]):
        old_alive = cache.alive
        logits, cache = model.decode(tokens[:, t:t+1], cache)
        direct = model(tokens[:, :t+1], mx.ones((1, t+1), dtype=mx.bool_), mx.array([9]), mx.array([[t]]))
        assert mx.allclose(logits, direct, atol=3e-6, rtol=3e-5).item()
        for previous, live in zip(old_alive, cache.alive):
            assert int(live.sum()) <= 4
            assert not bool(mx.any(live[:, :-1] & ~previous))
            assert bool(mx.all(live[:, -2:]))
    shorter = model(tokens[:, :6], mx.ones((1, 6), dtype=mx.bool_), mx.array([6]))
    longer = model(tokens, mx.ones(tokens.shape, dtype=mx.bool_), mx.array([9]))
    assert mx.allclose(shorter, longer[:, :6], atol=3e-6, rtol=3e-5).item()


def test_scored_future_loss_trains_scorer_and_backbone():
    model = small_model({"recent_window": 2, "memory_tokens": 1, "score_dim": 8})
    tokens = mx.array([[1, 10, 11, 12, 13, 14, 2]])
    def loss(m):
        logits = m(tokens, mx.ones(tokens.shape, dtype=mx.bool_), mx.array([7]), mx.array([[6]]))
        return nn.losses.cross_entropy(logits.astype(mx.float32), mx.array([[4]])).mean()
    value, grad = nn.value_and_grad(model, loss)(model)
    mx.eval(value, grad)
    gradients = dict(tree_flatten(grad))
    scorer = [v for k, v in gradients.items() if ".retention." in k]
    assert scorer and all(bool(mx.all(mx.isfinite(v))) for v in scorer)
    assert sum(float(mx.sum(mx.abs(v))) for v in scorer) > 1e-8
    assert float(mx.sum(mx.abs(gradients["embed.weight"]))) > 1e-8


def test_scored_priorities_are_assigned_once_and_remain_stable():
    model = small_model({"recent_window": 2, "memory_tokens": 2, "score_dim": 8})
    tokens = mx.array([[1, 10, 11, 12, 13, 14, 2]])
    _, cache = model.prefill(tokens, mx.ones(tokens.shape, dtype=mx.bool_),
                             mx.array([tokens.shape[1]]))
    before = [scores[:, :3] for scores in cache.retention_scores]
    _, cache = model.decode(mx.array([[260]]), cache)
    _, cache = model.decode(mx.array([[261]]), cache)
    mx.eval(before, cache.retention_scores)
    for old, scores in zip(before, cache.retention_scores, strict=True):
        assert mx.array_equal(old, scores[:, :3]).item()


def test_scored_restart_freezes_survivors():
    model = small_model({"recent_window": 2, "memory_tokens": 2})
    tokens = mx.array([[1, 10, 11, 12, 13, 14, 2]])
    session = DecodeSession(model, tokens, mx.ones(tokens.shape, dtype=mx.bool_), mx.array([7]), mode="restart-each")
    for _ in range(3):
        for a, b in zip(session.cache.alive, session.shadow.alive):
            assert mx.array_equal(a, b).item()
        session.advance(mx.array([[260]]))


def test_baseline_checkpoint_loads_exactly_and_can_initialize_scorer(tmp_path):
    from llmz.train import load_model_checkpoint
    baseline = small_model()
    path = tmp_path / 'baseline.npz'
    mx.savez(str(path), **{'model::'+k: v for k, v in tree_flatten(baseline.parameters())})
    path.with_suffix('.json').write_text('{"step": 6000}')
    model = small_model({'recent_window': 2, 'memory_tokens': 2})
    assert load_model_checkpoint(path, model)['step'] == 6000
    actual = dict(tree_flatten(model.parameters()))
    for name, value in tree_flatten(baseline.parameters()):
        assert mx.array_equal(value, actual[name]).item()
