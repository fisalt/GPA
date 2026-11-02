from dataclasses import replace
import pytest
import torch
from gpa import DenseGPA, GPAConfig
from gpa.state import tensors


@pytest.mark.parametrize("mode", ["gated", "moments", "none", "aligned", "rsa", "poe", "positive", "tanh", "marginal", "quadratic", "temperature"])
@pytest.mark.parametrize("levels", [1, 2, 4])
def test_packed_equivalence_and_gradients(mode, levels):
    torch.manual_seed(99)
    torch.set_num_threads(1)
    cfg = GPAConfig(heads=2, key_dim=4, value_dim=3, slots=(3,)*levels,
                     positional_mode="gated" if mode in ("aligned","rsa","poe","positive","tanh","marginal","quadratic","temperature") else mode, chunk_size=3,
                     shared_query_center=mode in ("aligned","positive","tanh","marginal","quadratic"),rsa_update=mode in ("rsa","poe","positive","tanh","marginal","quadratic"),
                     paper_product_read=mode in ("poe","positive","tanh","marginal","quadratic"),relation_marginal=mode=="marginal",
                     slot_feature="softmax" if mode=="positive" else "tanh" if mode=="tanh" else "linear",
                     balanced_bridge=mode in ("tanh","marginal","quadratic"),quadratic_features=mode=="quadratic",
                     learn_read_scale=mode=="temperature")
    reference = DenseGPA(cfg).double()
    packed = DenseGPA(replace(cfg, execution="packed")).double()
    packed.load_state_dict(reference.state_dict())
    xs = [torch.randn(2, 7, 2, d, dtype=torch.double, requires_grad=True) for d in (4, 4, 3)]
    ys = [x.detach().clone().requires_grad_() for x in xs]
    a, sa = reference(*xs)
    b, sb = packed(*ys)
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    for x, y in zip(tensors(sa), tensors(sb)):
        torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)
    a.square().sum().backward()
    b.square().sum().backward()
    for x, y in zip(xs, ys):
        torch.testing.assert_close(x.grad, y.grad, atol=1e-9, rtol=1e-9)
    for (name, x), (_, y) in zip(reference.named_parameters(), packed.named_parameters()):
        if x.grad is not None:
            assert y.grad is not None, name
            torch.testing.assert_close(x.grad, y.grad, atol=1e-9, rtol=1e-9, msg=name)


def test_full_hidden_control_packing_and_cached_gradients():
    torch.manual_seed(84)
    cfg=GPAConfig(heads=2,key_dim=4,value_dim=3,slots=(3,3),chunk_size=3,
                  control_dim=7,rsa_update=True,paper_product_read=True,relation_marginal=True)
    a=DenseGPA(cfg).double();b=DenseGPA(replace(cfg,execution="packed")).double()
    with torch.no_grad():
        for w in a.gate_weight:w.normal_(0,.1)
    b.load_state_dict(a.state_dict())
    inputs=[torch.randn(1,7,2,d,dtype=torch.double,requires_grad=True) for d in (4,4,3)]
    control=torch.randn(1,7,7,dtype=torch.double,requires_grad=True)
    expected,_=a(*inputs,control_input=control)
    prefix,state=b(*(x[:,:2] for x in inputs),control_input=control[:,:2])
    rest,state=b(*(x[:,2:] for x in inputs),state,control_input=control[:,2:])
    actual=torch.cat((prefix,rest),1)
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
    actual.square().sum().backward()
    assert control.grad.abs().sum()>0
    assert all(w.grad.abs().sum()>0 for w in b.gate_weight)
