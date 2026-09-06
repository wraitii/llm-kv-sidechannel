import numpy as np

from llmz.carriers import expand_batch
from llmz.transport import RecursiveCarrierPolicy, max_carrier_tokens


def make_batch(sources, prefix_lengths=None):
    """Packed rows: [BOS, *source, SEP, 2 target tokens, EOS]."""
    rows = len(sources)
    width = max(len(s) for s in sources) + 5
    x = np.zeros((rows, width), dtype=np.int32)
    valid = np.zeros_like(x, dtype=bool)
    out_pos = np.zeros((rows, 3), dtype=np.int32)
    for row, source in enumerate(sources):
        seq = [1, *source, 2, 50, 51, 3]
        x[row, :len(seq)] = seq
        valid[row, :len(seq)] = True
        out_pos[row] = [len(source) + 2, len(source) + 3, len(source) + 4]
    if prefix_lengths is None:
        prefix_lengths = np.array([len(s) + 2 for s in sources], dtype=np.int32)
    return {"x": x, "valid": valid,
            "source_tokens": np.array([len(s) for s in sources]),
            "prefix_lengths": prefix_lengths, "output_positions": out_pos}


def test_expand_batch_inserts_carriers_and_remaps_positions():
    policy = RecursiveCarrierPolicy(hidden_tokens=2, carrier_tokens=2,
                                    gap_tokens=0, group_size=8, depth=1)
    plan = policy.plan(np.array([6]), np.random.default_rng(0))
    # With h=2, g=0 the tiling is forced: blocks (0,2), (2,4), (4,6), each +2.
    assert plan.blocks[0] == [(0, 2, 2), (2, 4, 2), (4, 6, 2)]
    batch = make_batch([[10, 11, 12, 13, 14, 15]])
    out, spans, remap = expand_batch(batch, plan, policy, [900, 901])
    # [BOS 10 11 C0 C1 12 13 C0 C1 14 15 C0 C1 SEP 50 51 EOS]
    assert out["x"][0, :17].tolist() == [
        1, 10, 11, 900, 901, 12, 13, 900, 901, 14, 15, 900, 901, 2, 50, 51, 3]
    assert out["valid"][0, :17].tolist() == [True] * 17
    # cached position p maps to p + carriers inserted before it
    assert remap[0, [1, 2, 3, 6, 9, 10]].tolist() == [1, 2, 5, 10, 15, 16]
    assert remap[0, 7] == 13   # SEP shifts by all six carriers
    # output positions (target region) shift by the carrier total
    assert out["output_positions"][0].tolist() == [14, 15, 16]
    assert out["prefix_lengths"][0] == 14  # count: SEP expanded pos + 1
    # level-1 spans: block [1,3) hidden after its carriers end at 4
    assert spans[0, 0].tolist() == [1, 3, 4]
    assert spans[0, 1].tolist() == [5, 7, 8]
    assert spans[0, 2].tolist() == [9, 11, 12]


def test_expand_batch_deeper_levels_hide_runs_behind_meta_carriers():
    policy = RecursiveCarrierPolicy(hidden_tokens=1, carrier_tokens=1,
                                    gap_tokens=0, group_size=2, depth=2)
    plan = policy.plan(np.array([8]), np.random.default_rng(0))
    assert plan.blocks[0] == [(i, i + 1, 1) for i in range(8)]
    batch = make_batch([[10 + i for i in range(8)]])
    out, spans, _ = expand_batch(batch, plan, policy, [900])
    # sequence: BOS s0 C0 s1 C0 s2 C0 ... s7 C0 SEP...
    carrier_pos = [2 + 2 * i for i in range(8)]
    assert out["x"][0, carrier_pos].tolist() == [900] * 8
    # level 2: runs of two groups hide behind the next group's carrier
    # (the whole inter-group range, including the ordinary token between)
    deeper = spans[0][spans[0, :, 1] - spans[0, :, 0] != 1]
    assert deeper.tolist() == [[2, 5, 6], [6, 9, 10], [10, 13, 14]]


def test_expand_batch_without_blocks_is_identity_plus_padding():
    from llmz.transport import CarrierPlan
    policy = RecursiveCarrierPolicy(hidden_tokens=1, carrier_tokens=2,
                                    gap_tokens=0, group_size=4, depth=1)
    batch = make_batch([[10, 11], [12, 13, 14, 15]])
    plan = CarrierPlan([[], []])  # no blocks: pure identity remap
    out, spans, remap = expand_batch(batch, plan, policy, [900])
    assert spans.shape[1] == 0
    assert out["x"][0, :7].tolist() == batch["x"][0][:7].tolist()
    assert remap[0].min() >= 0


def test_carrier_bound_covers_sampled_plans():
    policy = RecursiveCarrierPolicy(hidden_tokens=[3, 5], carrier_tokens=[2, 3],
                                    gap_tokens=[0, 2], group_size=8, depth=2)
    bound = max_carrier_tokens({"kind": "recursive_carriers",
                                "hidden_tokens": [3, 5], "carrier_tokens": [2, 3],
                                "gap_tokens": [0, 2]}, 256)
    rng = np.random.default_rng(3)
    for length in rng.integers(1, 257, size=200):
        plan = policy.plan(np.array([length]), rng)
        total = sum(c for _, _, c in plan.blocks[0])
        assert total <= bound, (length, total, bound)


