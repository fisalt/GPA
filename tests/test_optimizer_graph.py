import copy
import pytest
import torch


def test_paper_lm_optimizer_graph_matches_eager():
    if not torch.cuda.is_available():pytest.skip("CUDA required")
    from gpa.retrieval import RetrievalConfig,RetrievalLM,make_batch
    from gpa.training_graph import GradientGraph
    torch.manual_seed(123)
    c=RetrievalConfig(memory_update="paper_dense")
    a=RetrievalLM(c).cuda();b=copy.deepcopy(a)
    oa=torch.optim.AdamW(a.parameters(),lr=.002,fused=True)
    ob=torch.optim.AdamW(b.parameters(),lr=.002,fused=True)
    ids,labels,_=make_batch(432143,8,64,1,device="cuda")
    def fn(model,x,y):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out=model(x)
            loss=torch.nn.functional.cross_entropy(out.float().flatten(0,1),y.flatten())
        return loss,out
    graph=GradientGraph(b,(ids,labels),lambda x,y:fn(b,x,y))
    for step in range(3):
        ids,labels,_=make_batch(432143+step,8,64,1,device="cuda")
        oa.zero_grad(set_to_none=True)
        la,_=fn(a,ids,labels);la.backward()
        lb,_=graph.replay(ids,labels)
        torch.testing.assert_close(la,lb,atol=1e-5,rtol=1e-5)
        numerator=denominator=0.
        for pa,pb in zip(a.parameters(),b.parameters()):
            if pa.grad is not None:
                numerator+=(pa.grad-pb.grad).square().sum()
                denominator+=pa.grad.square().sum()
        relative=(numerator/denominator.clamp_min(1e-12)).sqrt()
        print("optimizer graph step",step,"gradient relative L2",relative.item(),flush=True)
        assert relative<.015
        torch.nn.utils.clip_grad_norm_(a.parameters(),1.)
        torch.nn.utils.clip_grad_norm_(b.parameters(),1.)
        oa.step();ob.step()
        for pa,pb in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(pa,pb,atol=1e-4,rtol=2e-3)
