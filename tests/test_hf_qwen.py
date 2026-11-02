"""Optional HF integration tests; all fixtures are tiny random native Qwen LMs."""
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version
import copy
import pytest
import torch
import torch.nn.functional as F

if Version(torch.__version__.split("+")[0]) < Version("2.6"):
    pytest.skip("Optional Qwen integration requires torch>=2.6", allow_module_level=True)
try:
    if Version(version("transformers")) < Version("4.51"):
        pytest.skip("Qwen3 requires transformers>=4.51", allow_module_level=True)
except PackageNotFoundError:
    pytest.skip("Install the hf extra", allow_module_level=True)

from transformers import AutoConfig, AutoModelForCausalLM
from gpa.hf_qwen import GPAQwenForCausalLM, GPAQwenCache, LoRALinear
from gpa.finetune import configure_trainable


def tiny(kind="qwen3", rope=False, backend="torch", chunk=8):
    torch.manual_seed(91); torch.set_num_threads(2)
    options = dict(vocab_size=97, hidden_size=32 if kind=="qwen3" else 64,
                   intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                   num_key_value_heads=2, max_position_embeddings=512,
                   tie_word_embeddings=True, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    if kind=="qwen3": options["head_dim"] = 16  # 4*16 != hidden_size: real Qwen3 geometry.
    cfg = AutoConfig.for_model(kind, **options)
    native = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
    before = {n: p.detach().clone() for n,p in native.named_parameters()}
    model = GPAQwenForCausalLM.from_transformer(native, keep_rope=rope,
        gpa=dict(slots=(16,16), backend=backend, execution="staged", chunk_size=chunk),
        loss_chunk_size=7)
    for n,p in model.named_parameters():
        if n in before: assert torch.equal(p,before[n]),n
    assert len(model.model.layers)==2
    return model


@pytest.mark.parametrize("kind",["qwen2","qwen3"])
@pytest.mark.parametrize("rope",[False,True])
def test_conversion_no_softmax_chunked_loss_and_cache(kind,rope,monkeypatch):
    model = tiny(kind,rope).eval()
    ids = torch.randint(3,97,(2,37))
    def forbidden(*a,**kw): raise AssertionError("Transformer attention was called")
    monkeypatch.setattr(F,"scaled_dot_product_attention",forbidden)
    with torch.no_grad():
        full = model(ids,use_cache=False).logits
        loss = model(ids,labels=ids,use_cache=False).loss
        expected = F.cross_entropy(full[:,:-1].reshape(-1,97),ids[:,1:].reshape(-1))
        torch.testing.assert_close(loss,expected)
        cache=None;parts=[]
        for lo,hi in ((0,8),(8,9),(9,37)):
            output=model(ids[:,lo:hi],past_key_values=cache,use_cache=True)
            cache=output.past_key_values;parts.append(output.logits)
        assert isinstance(cache,GPAQwenCache) and cache.get_seq_length()==37
        torch.testing.assert_close(torch.cat(parts,1),full,atol=2e-5,rtol=2e-5)
    with pytest.raises(ValueError,match="Padding"):
        model(ids,attention_mask=torch.zeros_like(ids))


@pytest.mark.parametrize("kind",["qwen2","qwen3"])
def test_full_lora_checkpoint_auto_map_merge_and_generation(kind,tmp_path):
    model=tiny(kind,rope=True)
    audit=configure_trainable(model,rank=2)
    assert audit["lora_modules"]==14 and audit["softmax_attention_layers"]==0
    assert model.get_input_embeddings().weight.requires_grad
    ids=torch.randint(3,97,(2,25))
    model.gradient_checkpointing_enable({"use_reentrant":False})
    model.train();loss=model(ids,labels=ids,use_cache=False).loss
    assert torch.isfinite(loss);loss.backward()
    missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing,missing
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.model.layers[0].self_attn.gpa.inner_prefix[0].grad is not None
    for module in model.modules():
        if isinstance(module,LoRALinear):
            with torch.no_grad():module.lora_B.normal_(0,.002)
    model.eval();model.gradient_checkpointing_disable()
    with torch.no_grad():expected=model(ids,use_cache=False).logits
    folder=tmp_path/'full';model.save_pretrained(folder,safe_serialization=True)
    loaded,info=AutoModelForCausalLM.from_pretrained(folder,trust_remote_code=True,
        output_loading_info=True,local_files_only=True)
    assert not any(info.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")),info
    assert loaded.config.lora["rank"]==2
    loaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids,use_cache=False).logits,expected,atol=1e-6,rtol=1e-5)
        cached=loaded.generate(ids[:1,:9],max_new_tokens=3,do_sample=False,use_cache=True)
        uncached=loaded.generate(ids[:1,:9],max_new_tokens=3,do_sample=False,use_cache=False)
        assert torch.equal(cached,uncached)
        beam=loaded.generate(ids[:1,:9],max_new_tokens=3,num_beams=2,use_cache=True)
        assert beam.shape[1]>=10
    loaded.merge_lora()
    assert not any(type(m).__name__=="LoRALinear" for m in loaded.modules())
    with torch.no_grad():torch.testing.assert_close(loaded(ids,use_cache=False).logits,expected,atol=2e-5,rtol=2e-5)
    loaded.save_pretrained(tmp_path/'merged',safe_serialization=True)


def test_beam_reorder_moves_all_states_and_conv_tails():
    model=tiny().eval();ids=torch.randint(3,97,(2,17));order=torch.tensor([1,1,0])
    with torch.no_grad():
        cache=model(ids[:,:9],use_cache=True).past_key_values
        model._reorder_cache(cache,order)
        actual=model(ids.index_select(0,order)[:,9:],past_key_values=cache,use_cache=True).logits
        expected=model(ids.index_select(0,order),use_cache=False).logits[:,9:]
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)


def test_independent_final_layer_layout_roundtrip(tmp_path):
    model=tiny().eval()
    config=copy.deepcopy(model.config)
    config.gpa_last_slots=[8,8,8,8]
    varied=GPAQwenForCausalLM(config).eval()
    assert varied.model.layers[0].self_attn.gpa.config.slots==(16,16)
    assert varied.model.layers[-1].self_attn.gpa.config.slots==(8,8,8,8)
    ids=torch.randint(3,97,(1,17))
    with torch.no_grad():expected=varied(ids,use_cache=False).logits
    varied.save_pretrained(tmp_path/'varied')
    restored=GPAQwenForCausalLM.from_pretrained(tmp_path/'varied').eval()
    with torch.no_grad():torch.testing.assert_close(restored(ids,use_cache=False).logits,expected)


def test_cuda_fla_hf_forward_backward_cache():
    if not torch.cuda.is_available():pytest.skip("CUDA required")
    pytest.importorskip("fla")
    model=tiny(backend="fla",chunk=64).cuda()
    configure_trainable(model,rank=2)
    ids=torch.randint(3,97,(1,256),device="cuda")
    model.gradient_checkpointing_enable({"use_reentrant":False})
    with torch.autocast("cuda",dtype=torch.bfloat16):loss=model(ids,labels=ids,use_cache=False).loss
    loss.backward();assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval();model.gradient_checkpointing_disable()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        full=model(ids,use_cache=False).logits
        state=None;parts=[]
        for lo,hi in ((0,128),(128,129),(129,256)):
            out=model(ids[:,lo:hi],past_key_values=state,use_cache=True)
            state=out.past_key_values;parts.append(out.logits)
        error=(torch.cat(parts,1).float()-full.float()).norm()/full.float().norm()
        assert error<.05,error.item()
