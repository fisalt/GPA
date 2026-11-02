import torch
from gpa.features import QuadraticFeatures


def test_quadratic_norm_zero_and_gradcheck():
    torch.manual_seed(68);torch.set_num_threads(1)
    feature=QuadraticFeatures(2,4).double()
    x=torch.randn(1,3,2,4,dtype=torch.double,requires_grad=True)
    y=feature(x)
    torch.testing.assert_close(x.norm(dim=-1),y.norm(dim=-1),atol=1e-10,rtol=1e-10)
    assert torch.autograd.gradcheck(feature,(x,),atol=1e-5,rtol=1e-3)
    (y*torch.randn_like(y)).sum().backward()
    assert feature.left.grad.abs().sum()>0
    zero=torch.zeros_like(x,requires_grad=True)
    feature(zero).sum().backward()
    assert torch.isfinite(zero.grad).all()
