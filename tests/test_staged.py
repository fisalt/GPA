from dataclasses import replace
import pytest
import torch
from gpa import GPAConfig,DenseGPA
from gpa.state import tensors


@pytest.mark.parametrize("levels",[1,2,4])
@pytest.mark.parametrize("enhanced",[False,True])
def test_staged_exact_model_state_parameter_gradients_and_cached_calls(levels,enhanced):
    torch.set_num_threads(1);torch.manual_seed(71)
    dv=3 if enhanced else 4
    cfg=GPAConfig(heads=2,key_dim=4,value_dim=dv,slots=(4,)*levels,chunk_size=4,
        execution="packed",rsa_update=enhanced,paper_product_read=enhanced,
        balanced_bridge=enhanced,quadratic_features=enhanced,learn_read_scale=enhanced,
        relation_marginal=enhanced,control_dim=6 if enhanced else 0)
    a=DenseGPA(cfg).double();b=DenseGPA(replace(cfg,execution="staged")).double()
    b.load_state_dict(a.state_dict())
    xs=[torch.randn(1,32,2,d,dtype=torch.double,requires_grad=True) for d in (4,4,dv)]
    ys=[x.detach().clone().requires_grad_() for x in xs]
    ctrl=torch.randn(1,32,6,dtype=torch.double,requires_grad=True) if enhanced else None
    ctrl2=ctrl.detach().clone().requires_grad_() if enhanced else None
    ya,sa=a(*xs,control_input=ctrl);yb,sb=b(*ys,control_input=ctrl2)
    torch.testing.assert_close(ya,yb,atol=1e-10,rtol=1e-10)
    for u,v in zip(tensors(sa),tensors(sb)):torch.testing.assert_close(u,v,atol=1e-10,rtol=1e-10)
    def loss(y,s):return y.square().sum()+.01*sum(v.square().sum() for v in tensors(s))
    loss(ya,sa).backward();loss(yb,sb).backward()
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=1e-8,rtol=1e-8,msg=name)
    for u,v in zip(xs,ys):torch.testing.assert_close(u.grad,v.grad,atol=1e-8,rtol=1e-8)
    if enhanced:torch.testing.assert_close(ctrl.grad,ctrl2.grad,atol=1e-8,rtol=1e-8)
    with torch.no_grad():
        prefix,_=b(*(x[:,:23] for x in ys),control_input=ctrl2[:,:23] if enhanced else None)
        torch.testing.assert_close(prefix,yb[:,:23],atol=1e-10,rtol=1e-10)
        state=None;pieces=[]
        for lo,hi in ((0,16),(16,17),(17,28),(28,32)):
            y,state=b(*(x[:,lo:hi] for x in ys),state,control_input=ctrl2[:,lo:hi] if enhanced else None)
            pieces.append(y)
        torch.testing.assert_close(torch.cat(pieces,1),yb,atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize("levels,d,dv",[(2,32,32),(4,32,32),(2,64,32)])
def test_staged_cuda_forward_backward_and_cache(levels,d,dv):
    if not torch.cuda.is_available():pytest.skip("CUDA required")
    torch.manual_seed(72)
    cfg=GPAConfig(heads=2,key_dim=d,value_dim=dv,slots=(d,)*levels,chunk_size=64,
        backend="fla",execution="triton",rsa_update=True,paper_product_read=True,
        balanced_bridge=True,quadratic_features=True,beta_init=.6,read_scale=8.)
    a=DenseGPA(cfg).cuda();b=DenseGPA(replace(cfg,execution="staged")).cuda()
    b.load_state_dict(a.state_dict())
    xs=[torch.randn(1,512,2,dim,device="cuda",dtype=torch.bfloat16,requires_grad=True) for dim in (d,d,dv)]
    ys=[x.detach().clone().requires_grad_() for x in xs]
    with torch.autocast("cuda",dtype=torch.bfloat16):
        ya,_=a(*xs);yb,_=b(*ys)
    error=(ya.float()-yb.float()).norm()/ya.float().norm()
    assert error<.06,error.item()
    ya.float().square().mean().backward();yb.float().square().mean().backward()
    for x,y in zip(xs,ys):
        e=(x.grad.float()-y.grad.float()).norm()/x.grad.float().norm().clamp_min(1e-9)
        assert e<.15,e.item()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        prefix,_=b(*(x[:,:509] for x in ys))
        pe=(prefix.float()-yb[:,:509].float()).norm()/yb[:,:509].float().norm()
        assert pe<.06,pe.item()
        state=None;pieces=[]
        for lo,hi in ((0,256),(256,321),(321,512)):
            z,state=b(*(x[:,lo:hi] for x in ys),state);pieces.append(z)
        e=(torch.cat(pieces,1).float()-yb.float()).norm()/yb.float().norm()
        assert e<.06,e.item()
    print({"levels":levels,"forward_relative_l2":error.item(),"cache_relative_l2":e.item(),"prefix_relative_l2":pe.item()})
