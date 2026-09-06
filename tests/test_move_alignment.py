import numpy as np
import pytest

from llmz.move_alignment import move_boundaries, sample_transport
from llmz.transport import policy_from_config, max_carrier_tokens
from llmz.tokenizer import PairTokenizer, BytesTokenizer, BPETokenizer, train_bpe
from llmz.inference import prepare_prefix
from llmz.experiment import training_batch


def tokenizer():
    return PairTokenizer(BytesTokenizer(), BytesTokenizer(), carrier_vocab=2)


def test_boundaries_include_complete_promotions_and_separator():
    tok = tokenizer()
    text = 'e2e4 e7e5 a7a8q'
    assert move_boundaries(tok.encode_source(text), tok).tolist() == [0, 5, 10, 15]
    with pytest.raises(ValueError, match='complete UCI'):
        move_boundaries(tok.encode_source('e2e4 a7a'), tok)


def test_recursive_spans_never_split_moves_at_any_depth():
    tok = tokenizer()
    source = tok.encode_source(' '.join(['e2e4', 'e7e5', 'g1f3', 'a7a8q'] * 10))
    edges = set(move_boundaries(source, tok).tolist())
    policy = policy_from_config(dict(kind='recursive_blocks', alignment='move',
                                    hidden_moves=[1, 3], survivor_moves=[1, 2],
                                    gap_moves=[0, 1], group_size=3, depth=3))
    spans = sample_transport(policy, [source], tok, np.random.default_rng(3))[0]
    for start, end, visible in spans:
        if start >= 0:
            assert start in edges and end in edges and visible + 1 in edges
    # First block's successor move has completely arrived before eviction.
    assert spans[0, 2] >= spans[0, 1] + 3


def test_move_aligned_carriers_and_native_inference_match_training():
    tok = tokenizer()
    source = tok.encode_source('e2e4 e7e5 a7a8q g8f6')
    cfg = dict(kind='recursive_carriers', alignment='move', hidden_moves=1,
               carrier_tokens=2, gap_moves=0, group_size=3, depth=2)
    policy = policy_from_config(cfg)
    plan = sample_transport(policy, [source], tok, np.random.default_rng(0))
    assert [(start, end) for start, end, _ in plan.blocks[0]] == [(0,5),(5,10),(10,16),(16,20)]
    assert sum(c for _, _, c in plan.blocks[0]) <= max_carrier_tokens(cfg, len(source))
    ids = np.array([[1, *source, 2]])
    batch = dict(x=ids, valid=np.ones(ids.shape, bool), source_tokens=np.array([len(source)]),
                 prefix_lengths=np.array([ids.shape[1]]), output_positions=np.array([[ids.shape[1]-1]]))
    trained = training_batch(batch, policy, tok, np.random.default_rng(0))
    prefix, valid, _, spans = prepare_prefix(tok, [{'asm':tok.decode_source(source)}], 64,
                                           policy, transport=True)
    np.testing.assert_array_equal(np.asarray(prefix), trained['x'])
    np.testing.assert_array_equal(np.asarray(spans), trained['transport_spans'])


def test_transport_configs_require_explicit_move_units():
    with pytest.raises(ValueError):
        policy_from_config({'kind':'recursive_blocks'})
    with pytest.raises(ValueError):
        policy_from_config({'kind':'recursive_blocks', 'alignment':'move',
                            'hidden_moves':2, 'hidden_tokens':4})


def test_chess_bpe_move_offsets(tmp_path):
    import json
    text = 'e2e4 e7e5 g1f3 b8c6 a7a8q'
    data = tmp_path/'rows.jsonl'
    data.write_text(json.dumps({'history':text})+'\n')
    path=tmp_path/'tokenizer.json'
    train_bpe(data, 'history', path, 280, 'chess')
    tok = PairTokenizer(BPETokenizer(path), BytesTokenizer())
    source=tok.encode_source(text)
    edges=move_boundaries(source,tok)
    assert len(edges)==6
    for i, move in enumerate(text.split()):
        assert tok.decode_source(source[edges[i]:edges[i+1]]).strip()==move
