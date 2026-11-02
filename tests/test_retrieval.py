import random
import pytest
from dataclasses import replace
import torch
from gpa.retrieval import make_example, make_batch, TASKS, RetrievalLM, RetrievalConfig


def test_overwrite_v2_always_queries_updated_key():
    from gpa.retrieval import PAIR,QUERY,VAL,END,ANSWER
    for seed in range(20):
        ids,labels,groups=make_example(random.Random(seed),128,"overwrite",8,data_version=2)
        last_pair=max(i for i,x in enumerate(ids) if x==PAIR)
        query=ids.index(QUERY)
        assert ids[last_pair+1]==ids[query+1]
        assert labels[groups[0][0]]==ids[last_pair+3]
        old_records=[i for i,x in enumerate(ids[:last_pair]) if x==PAIR and ids[i+1]==ids[query+1]]
        assert len(old_records)==1 and ids[old_records[0]+3]!=ids[last_pair+3]


def test_synthetic_shapes_labels_and_independent_seeds():
    for task in TASKS:
        ids, labels, groups = make_example(random.Random(10), 128, task, 4)
        assert len(ids) == len(labels) == 128
        assert all(ids[pos+1] == labels[pos] for group in groups for pos in group)
        assert sum(x != -100 for x in labels) == sum(map(len, groups))
    train, _, _ = make_batch(100, 4, 128, 4)
    test, _, _ = make_batch(900000, 4, 128, 4)
    assert not torch.equal(train, test)


def test_overwrite_v3_requires_key_and_latest_value():
    from gpa.retrieval import PAIR,QUERY
    last_key_queries=0
    for seed in range(100):
        ids,labels,groups=make_example(random.Random(seed),256,"overwrite",8,data_version=3)
        query=ids[ids.index(QUERY)+1]
        pairs=[i for i,x in enumerate(ids) if x==PAIR]
        matches=[i for i in pairs if ids[i+1]==query]
        assert len(pairs)==12 and len(matches)==2
        assert ids[matches[0]+3]!=ids[matches[1]+3]
        assert labels[groups[0][0]]==ids[matches[-1]+3]
        last_key_queries+=ids[pairs[-1]+1]==query
    assert 10<last_key_queries<45


@pytest.mark.parametrize("gated",[False,True])
def test_tiny_retrieval_model_causal_and_backward_cpu(gated):
    torch.set_num_threads(1)
    cfg = RetrievalConfig(width=16, heads=2, layers=2, slots=(3, 3), chunk_size=8,
                          backend="torch", execution="packed",gated_output=gated,full_controls=gated,
                          context_controls=gated,contextual_output=gated,temporal_qk_init=gated,
                          key_dim=12 if gated else 0)
    model = RetrievalLM(cfg)
    if gated:
        with torch.no_grad():
            for block in model.blocks:
                for weight in block.attention.gate_weight:weight.normal_(0,.01)
    ids = torch.randint(0, 256, (1, 16))
    a = model(ids)
    b = model(ids[:, :9])
    torch.testing.assert_close(a[:, :9], b, atol=1e-5, rtol=1e-5)
    a.square().mean().backward()
    assert torch.isfinite(model.embedding.weight.grad).all()
    if gated:
        assert model.blocks[0].output_gate.weight.grad.abs().sum()>0
        assert model.blocks[0].head_norm_weight.grad.abs().sum()>0
        assert torch.isfinite(model.blocks[0].control_conv.weight.grad).all()
        assert model.blocks[0].control_conv.weight.grad.abs().sum()>0
