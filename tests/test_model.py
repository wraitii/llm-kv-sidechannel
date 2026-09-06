import mlx.core as mx
import mlx.nn as nn

from llmz.model import PrefixLM


def test_forward_and_backward_with_grouped_query_attention():
    model = PrefixLM(source_vocab_size=100, target_vocab_size=204,
                     d_model=64, layers=2, heads=4,
                     kv_heads=2, max_length=16, dtype=mx.float32)
    x = mx.array([[1, 10, 11, 2, 20, 21], [1, 12, 2, 22, 0, 0]])
    y = mx.array([[10, 11, 2, 20, 21, 3], [12, 2, 22, 3, 0, 0]])
    valid = mx.array([[True] * 6, [True, True, True, True, False, False]])
    prefix_lengths = mx.array([4, 3])
    loss_mask = mx.array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0]])

    def loss_fn(active_model):
        logits = active_model(x, valid, prefix_lengths)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.sum(losses * loss_mask) / mx.sum(loss_mask)

    loss, gradients = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, gradients)
    assert mx.isfinite(loss).item()


def test_logits_only_cover_target_vocabulary():
    model = PrefixLM(source_vocab_size=300, target_vocab_size=40,
                     d_model=32, layers=1, heads=4, kv_heads=2,
                     max_length=8, dtype=mx.float32)
    logits = model(mx.array([[1, 20, 2]]), mx.array([[True, True, True]]),
                   mx.array([3]))
    assert logits.shape == (1, 3, 40)
    selected = model(mx.array([[1, 20, 2]]), mx.array([[True, True, True]]),
                     mx.array([3]), mx.array([[0, 2]]))
    assert selected.shape == (1, 2, 40)


def test_external_embeddings_match_token_forward():
    mx.random.seed(3)
    model = PrefixLM(source_vocab_size=100, target_vocab_size=40,
                     d_model=32, layers=1, heads=4, kv_heads=2,
                     max_length=8, dtype=mx.float32)
    tokens = mx.array([[1, 20, 2, 100]])
    valid = mx.ones(tokens.shape, dtype=mx.bool_)
    prefix_lengths = mx.array([3])
    positions = mx.array([[2, 3]])
    token_logits = model(tokens, valid, prefix_lengths, positions)
    embedding_logits = model.from_embeddings(
        model.embed(tokens), valid, prefix_lengths, positions)
    mx.eval(token_logits, embedding_logits)
    assert mx.allclose(token_logits, embedding_logits).item()


def test_pause_embedding_is_not_a_target_class():
    model = PrefixLM(source_vocab_size=100, target_vocab_size=40,
                     d_model=32, layers=1, heads=4, kv_heads=2,
                     max_length=9, dtype=mx.float32, pause_token=True)
    logits = model(mx.array([[1, 20, 2, 136]]),
                   mx.array([[True, True, True, True]]), mx.array([4]))
    assert model.embed.weight.shape == (137, 32)
    assert logits.shape == (1, 4, 40)


def test_cached_generation_matches_full_forward():
    mx.random.seed(7)
    model = PrefixLM(source_vocab_size=100, target_vocab_size=40,
                     d_model=32, layers=2, heads=4, kv_heads=2,
                     max_length=12, dtype=mx.float32)
    prefix = mx.array([[1, 10, 11, 2]])
    valid = mx.ones(prefix.shape, dtype=mx.bool_)
    cached_logits, cache = model.prefill(prefix, valid, mx.array([4]))
    full_logits = model(prefix, valid, mx.array([4]), mx.array([[3]]))
    mx.eval(cached_logits, full_logits, cache)
    assert mx.allclose(cached_logits, full_logits, rtol=1e-5, atol=1e-5).item()

    tokens = [1, 10, 11, 2]
    # Model IDs 100 and 101 correspond to target-local IDs 4 and 5.
    for token in (100, 101, 102):
        tokens.append(token)
        cached_logits, cache = model.decode(mx.array([[token]]), cache)
        sequence = mx.array([tokens])
        full_logits = model(sequence, mx.ones(sequence.shape, dtype=mx.bool_),
                            mx.array([4]), mx.array([[len(tokens) - 1]]))
        mx.eval(cached_logits, full_logits, cache)
        assert mx.allclose(cached_logits, full_logits,
                           rtol=1e-5, atol=1e-5).item()


def test_attention_capture_matches_full_forward():
    """Captured attention maps must reproduce the fast-path forward output."""
    model = PrefixLM(source_vocab_size=100, target_vocab_size=204,
                     d_model=64, layers=2, heads=4, kv_heads=2,
                     max_length=16, dtype=mx.float32)
    x = mx.array([[1, 10, 11, 2, 20, 21, 22]])
    valid = mx.array([[True] * 7])
    prefix_lengths = mx.array([4])

    direct = model(x, valid, prefix_lengths)
    hidden, maps = model.attention_maps(x, valid, prefix_lengths)
    captured = model._output(hidden)
    mx.eval(direct, hidden, captured, maps)
    assert len(maps) == 2
    for probs in maps:
        assert probs.shape == (1, 4, 7, 7)
        assert mx.allclose(probs.sum(axis=-1), mx.ones((1, 4, 7))).item()
    assert mx.allclose(direct, captured).item()
