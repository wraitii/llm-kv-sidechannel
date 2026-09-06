import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from llmz.data import batch_examples
from llmz.model import PrefixLM
from llmz.readouts import prepare_readouts, shared_readout_loss
from llmz.tokenizer import PairTokenizer, BytesTokenizer


def make_model(scored=None):
    mx.random.seed(41)
    return PrefixLM(260, 260, 32, 2, 4, 2, 160, dtype=mx.float32,
                    attention_mode="causal", scored_eviction=scored)


def raw_batch(tok):
    histories = ["e2e4 e7e5 g1f3 b8c6 f1b5 a7a6", "d2d4 d7d5 c1f4 g8f6"]
    examples = []
    for text in histories:
        source = tok.encode_source(text)
        target = tok.encode_target_local("8/8/8/8/8/8/8/8 w - - 0 1")
        sequence = [1, *source, 2, *tok.target_local_to_model(target), 3]
        examples.append((sequence, target, len(source) + 1, 1,
                         len(text), len(source), len(target) + 1))
    return batch_examples(examples)


def independent_loss(model, prepared):
    total = mx.array(0.0)
    count = 0
    rows, points, _ = prepared["branch_x"].shape
    for row in range(rows):
        for point in range(points):
            length = int(prepared["points"][row, point])
            source = prepared["source_x"][row, :length]
            answer = prepared["branch_x"][row, point]
            active = prepared["branch_valid"][row, point]
            tokens = np.concatenate([source, answer[active]])
            valid = mx.ones((1, len(tokens)), dtype=mx.bool_)
            spans = mx.array(prepared["spans"][row:row + 1])
            output = mx.array(np.arange(length, len(tokens))[None])
            logits = model(mx.array(tokens[None]), valid, mx.array([length]),
                           output, spans)
            labels = mx.array(prepared["branch_y"][row, point, active][None])
            total += nn.losses.cross_entropy(logits.astype(mx.float32), labels).sum()
            count += int(active.sum())
    return total / count


def test_three_shared_readouts_match_independent_prefixes():
    tok = PairTokenizer(BytesTokenizer(), BytesTokenizer())
    raw = raw_batch(tok)
    for scored in (None, {"recent_window": 8, "memory_tokens": 3}):
        model = make_model(scored)
        prepared = prepare_readouts(raw, None, tok, np.random.default_rng(1),
                                    np.random.default_rng(2), points=3,
                                    min_plies=2)
        shared = shared_readout_loss(
            model, *[mx.array(prepared[key]) for key in
                     ("source_x", "source_valid", "spans", "points",
                      "branch_x", "branch_y", "branch_valid")])
        separate = independent_loss(model, prepared)
        mx.eval(shared, separate)
        assert mx.allclose(shared, separate, atol=3e-6, rtol=3e-5).item()


def test_readouts_always_include_endpoint_and_distinct_earlier_points():
    tok = PairTokenizer(BytesTokenizer(), BytesTokenizer())
    raw = raw_batch(tok)
    prepared = prepare_readouts(raw, None, tok, np.random.default_rng(1),
                                np.random.default_rng(2), points=3,
                                min_plies=2)
    for row, length in enumerate(raw["source_tokens"]):
        assert prepared["points"][row, -1] == length + 1
        assert np.all(np.diff(prepared["points"][row]) > 0)


def test_swapped_support_counterfactual_is_differentiable():
    tok = PairTokenizer(BytesTokenizer(), BytesTokenizer())
    model = make_model({"recent_window": 8, "memory_tokens": 3})
    prepared = prepare_readouts(
        raw_batch(tok), None, tok, np.random.default_rng(1),
        np.random.default_rng(2), points=3, min_plies=2)
    arrays = [mx.array(prepared[key]) for key in
              ("source_x", "source_valid", "spans", "points",
               "branch_x", "branch_y", "branch_valid")]
    negative = mx.array([0, 2, 0, 17])

    def loss_fn(active_model):
        return shared_readout_loss(active_model, *arrays, negative=negative,
                                   negative_weight=0.1, negative_swaps=2,
                                   negative_alternatives=4,
                                   return_metrics=True)

    (loss, delta, better, count), gradients = nn.value_and_grad(model, loss_fn)(model)
    scorer_gradients = [value for name, value in tree_flatten(gradients)
                        if ".retention." in name]
    mx.eval(loss, delta, better, count, scorer_gradients)
    assert np.isfinite(float(loss.item()))
    assert scorer_gradients
    assert float(count.item()) == 4
    assert 0 <= float(better.item()) <= 4
    assert np.isfinite(float(delta.item()))
    assert any(float(mx.sum(mx.abs(value)).item()) > 0
               for value in scorer_gradients)


def test_scored_full_rows_bypass_eviction_for_every_readout():
    tok = PairTokenizer(BytesTokenizer(), BytesTokenizer())
    model = make_model({"recent_window": 8, "memory_tokens": 3})
    prepared = prepare_readouts(
        raw_batch(tok), None, tok, np.random.default_rng(1),
        np.random.default_rng(2), points=3, min_plies=2,
        full_attention_share=1.0)
    assert prepared["full_rows"].all()
    arrays = [mx.array(prepared[key]) for key in
              ("source_x", "source_valid", "spans", "points",
               "branch_x", "branch_y", "branch_valid")]
    full_loss = shared_readout_loss(
        model, *arrays, full_rows=mx.array(prepared["full_rows"]))
    retentions = [block.retention for block in model.blocks]
    for block in model.blocks:
        block.retention = None
    expected = independent_loss(model, prepared)
    for block, retention in zip(model.blocks, retentions, strict=True):
        block.retention = retention
    mx.eval(full_loss, expected)
    assert mx.allclose(full_loss, expected, atol=3e-6, rtol=3e-5).item()
