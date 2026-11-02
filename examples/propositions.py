"""Numerical illustrations of P1--P5, without large-model accuracy claims."""
import json
import math
import torch
from gpa.ops import rope
from gpa.propositions import (ExactKVCache, DecoupledRoPESequence, PositionalMomentCache,
                              analyze_interlayer_similarity, ResidualCodebook)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    keys = torch.randn(3, 8, dtype=torch.double)
    values = torch.randn(3, 4, dtype=torch.double)
    ids = torch.tensor([0, 1, 0, 2, 1, 0])
    query = torch.randn(8, dtype=torch.double)
    cache = ExactKVCache()
    for i in ids:
        cache.append(keys[i], values[i])
    dense = (query @ keys[ids].T / math.sqrt(8)).softmax(-1) @ values[ids]
    positions = torch.arange(len(ids))
    separated = DecoupledRoPESequence(keys, ids, positions)
    p1 = (dense-cache.read(query)).abs().max().item()
    p2 = (separated.reconstruct()-rope(keys[ids], positions)).abs().max().item()
    small_keys = torch.zeros_like(keys)
    small_keys[:, -2:] = keys[:, -2:] * .3  # low-frequency, small-angle fixture
    moments = PositionalMomentCache()
    for t, i in enumerate(ids):
        moments.append(small_keys[i], values[i], t)
    qr = rope(small_keys[0], len(ids))
    kr = rope(small_keys[ids], positions)
    true = (qr @ kr.T / math.sqrt(8)).softmax(-1) @ values[ids]
    p3 = (true-moments.read(small_keys[0], len(ids))).abs().max().item()
    report = analyze_interlayer_similarity(torch.eye(3), torch.eye(3))
    c = torch.tensor([[0., 0.], [1., 0.]])
    codebook = ResidualCodebook([c, c])
    x = torch.tensor([[1.8, .2]])
    reconstructed, _, residual = codebook(x)
    print(json.dumps({
        "p1_joint_KV_max_abs_error": p1, "p1_types_vs_tokens": [len(cache.keys), len(ids)],
        "p2_reconstruction_max_abs_error": p2,
        "p2_metadata_elements_NOT_constant_memory": separated.metadata_elements(),
        "p3_controlled_small_angle_max_abs_error": p3,
        "p4_assumptions_certified": report.bound_applicable,
        "p4_conditional_capacity": report.conditional_capacity_bound,
        "p5_index_tuples_vs_distinct_sums": [codebook.index_tuple_capacity, codebook.audit_distinct_sums()],
        "p5_telescoping_error": (x-reconstructed-residual[-1]).abs().max().item(),
    }, indent=2))


if __name__ == "__main__":
    main()
