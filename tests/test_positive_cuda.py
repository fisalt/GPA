"""Nonlinear feature/full-control extension versus actual recurrent equations."""
from dataclasses import replace
import pytest
import torch
from gpa import DenseGPA,GPAConfig


@pytest.mark.parametrize("feature",["softmax","tanh","quadratic","wide","temperature"])
def test_positive_full_control_fla_gradients_and_cache(feature):
    if not torch.cuda.is_available():pytest.skip("CUDA required")
    torch.manual_seed(35)
    dim=64 if feature=="wide" else 32
    cfg=GPAConfig(heads=2,key_dim=dim,value_dim=32,slots=(32,32),chunk_size=64,
        control_dim=64,rsa_update=True,paper_product_read=True,shared_query_center=True,
        slot_feature="linear" if feature in ("quadratic","wide","temperature") else feature,
        balanced_bridge=feature in ("tanh","quadratic","wide","temperature"),quadratic_features=feature in ("quadratic","wide","temperature"),
        learn_read_scale=feature=="temperature")
    a=DenseGPA(cfg).cuda()
    with torch.no_grad():
        for w in a.gate_weight:w.normal_(0,.03)
    b=DenseGPA(replace(cfg,backend="fla",execution="triton")).cuda()
    b.load_state_dict(a.state_dict())
    xs=[torch.randn(1,96,2,d,device="cuda",dtype=torch.bfloat16,requires_grad=True) for d in (dim,dim,32)]
    ctrl=torch.randn(1,96,64,device="cuda",requires_grad=True)
    ys=[x.detach().clone().requires_grad_() for x in xs]
    ctrl2=ctrl.detach().clone().requires_grad_()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        expected,_=a(*xs,control_input=ctrl)
        actual,_=b(*ys,control_input=ctrl2)
    assert (expected.float()-actual.float()).norm()/expected.float().norm()<.06
    expected.float().square().mean().backward();actual.float().square().mean().backward()
    for x,y in zip([*xs,ctrl],[*ys,ctrl2]):
        assert (x.grad.float()-y.grad.float()).norm()/x.grad.float().norm().clamp_min(1e-9)<.15
    assert b.relation_prefix[0].grad.abs().sum()>0
    if feature=="temperature":
        assert (a.read_log_scale.grad-b.read_log_scale.grad).norm()/a.read_log_scale.grad.norm().clamp_min(1e-9)<.15
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        p,state=b(*(x[:,:31] for x in ys),control_input=ctrl2[:,:31])
        r,state=b(*(x[:,31:] for x in ys),state,control_input=ctrl2[:,31:])
    cat=torch.cat((p,r),1)
    assert (cat.float()-actual.float()).norm()/actual.float().norm()<.06
