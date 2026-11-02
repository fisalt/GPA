import pytest
import torch
from gpa.paper_dense import PaperConfig,PaperDenseGPA,decayed_counts,moment_scan
from gpa.state import tensors


def test_count_and_centroid_recurrence():
    torch.manual_seed(2)
    p=torch.rand(2,7,2,3,dtype=torch.double)
    g=-torch.rand(2,7,2,dtype=torch.double)*.01
    n0=torch.rand(2,2,3,dtype=torch.double)
    n=n0; xs=[]
    for i in range(7):
        n=n*g[:,i].exp()[...,None]+p[:,i];xs.append(n)
    torch.testing.assert_close(decayed_counts(p,g,n0),torch.stack(xs,1))
    vg=-p*2
    n=n0;xs=[]
    for i in range(7):
        n=n*vg[:,i].exp()+p[:,i];xs.append(n)
    torch.testing.assert_close(decayed_counts(p,vg,n0),torch.stack(xs,1))


@pytest.mark.parametrize("read",["local","transport","slot","strong"])
def test_dense_paper_cache_gradients_and_causality(read):
    torch.set_num_threads(1);torch.manual_seed(3)
    model=PaperDenseGPA(PaperConfig(heads=2,key_dim=4,value_dim=3,slots=(3,3,3),chunk_size=3,
        dense_read="local" if read=="local" else "transport",slot_decay=read in ("slot","strong"),strong_relations=read=="strong")).double()
    xs=[torch.randn(2,7,2,d,dtype=torch.double,requires_grad=True) for d in (4,4,3)]
    y,state=model(*xs)
    a,sa=model(*(x[:,:2] for x in xs))
    b,sb=model(*(x[:,2:] for x in xs),sa)
    torch.testing.assert_close(y,torch.cat((a,b),1),atol=1e-10,rtol=1e-10)
    for s,t in zip(tensors(state),tensors(sb)):
        torch.testing.assert_close(s,t,atol=1e-10,rtol=1e-10)
    prefix,_=model(*(x[:,:4] for x in xs))
    torch.testing.assert_close(y[:,:4],prefix,atol=1e-10,rtol=1e-10)
    assert len(state.relations)==6 and (2,0) in model.edges
    y.square().sum().backward()
    for name,p in model.named_parameters():
        assert p.grad is not None,name
        assert torch.isfinite(p.grad).all(),name


@pytest.mark.parametrize("normalize",[True,False])
def test_dense_paper_input_gradcheck(normalize):
    torch.set_num_threads(1);torch.manual_seed(4)
    model=PaperDenseGPA(PaperConfig(heads=1,key_dim=2,value_dim=2,slots=(2,2),chunk_size=2,normalize_features=normalize)).double()
    xs=tuple(torch.randn(1,2,1,2,dtype=torch.double,requires_grad=True) for _ in range(3))
    assert torch.autograd.gradcheck(lambda *x:model(*x)[0],xs,atol=2e-4,rtol=2e-3)


@pytest.mark.parametrize("slot",[False,True])
def test_dense_paper_fla_parity(slot):
    if not torch.cuda.is_available():pytest.skip("CUDA required")
    from dataclasses import replace
    c=PaperConfig(heads=2,key_dim=32,value_dim=32,slots=(16,16),chunk_size=64,slot_decay=slot)
    a=PaperDenseGPA(c).cuda();b=PaperDenseGPA(replace(c,backend="fla")).cuda();b.load_state_dict(a.state_dict())
    x=[torch.randn(1,128,2,32,device="cuda",dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    z=[v.detach().clone().requires_grad_() for v in x]
    with torch.autocast("cuda",dtype=torch.bfloat16):
        y,sa=a(*x);v,sb=b(*z)
    error=(y.float()-v.float()).norm()/y.float().norm()
    assert error<.04,error
    y.float().square().mean().backward();v.float().square().mean().backward()
    for s,t in zip(x,z):
        assert (s.grad.float()-t.grad.float()).norm()/s.grad.float().norm().clamp_min(1e-8)<.1
