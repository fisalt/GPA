"""Inspect observed-token write rates and dense-R effective rank; no oracle writes."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import importlib.util
import subprocess
import sys
from pathlib import Path
import json
import torch
from gpa.retrieval import RetrievalConfig, RetrievalLM, make_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--output", default="diagnostics.json")
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--task", choices=["single","uuid4","mqar","overwrite"])
    p.add_argument("--data-version",type=int,choices=[1,2,3],default=3)
    p.add_argument("--ablate-relation-read",action="store_true")
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg=RetrievalConfig(**data["config"])
    if args.device=="cpu":cfg=replace(cfg,backend="torch",execution="packed")
    model = RetrievalLM(cfg).to(args.device).eval()
    model.load_state_dict(data["model"])
    ids, labels, _ = make_batch(901111, 8, args.length, 8 if args.length >= 256 else 4,
                               task=args.task,device=args.device,data_version=args.data_version)
    masks = {"noise": ids >= 128, "values": (ids >= 64) & (ids < 80),
             "keys": (ids >= 16) & (ids < 48), "markers": ids < 16,
             "answer_prediction_positions": labels != -100}
    reading=(ids==4).cumsum(1)>0
    masks.update(record_values=masks["values"] & ~reading,
                 generated_or_teacher_answer_values=masks["values"] & reading,
                 read_phase=reading,storage_phase=~reading)
    report, heatmaps, route_maps, gate_maps = {}, [], [], []
    handles,controls,originals = [],{},[]
    for index, block in enumerate(model.blocks):
        if not hasattr(block, "attention"):
            continue
        if model.config.gated_output:
            def read_gate_hook(module,inputs,output,index=index):
                gate=torch.nn.functional.silu(output).reshape(ids.shape[0],ids.shape[1],model.config.heads,-1)
                positions=torch.nonzero(labels[0]!=-100,as_tuple=True)[0]
                strength=gate[0].index_select(0,positions).float().abs().mean(-1)
                report.setdefault(f"layer_{index}",{})["read_gate_magnitude_by_answer_order_first_example"]=strength.cpu().tolist()
                gate_maps.append(strength.cpu().tolist())
            handles.append(block.output_gate.register_forward_hook(read_gate_hook))

        original=block.attention.forward
        originals.append((block.attention,original))
        def capture(*inputs, _original=original, _index=index, **kwargs):
            controls[_index]=kwargs.get("control_input")
            output,state,diag=_original(*inputs,return_diagnostics=True,**kwargs)
            routes=diag["routes"]
            if not isinstance(routes,torch.Tensor):routes=torch.stack(routes,2)
            mass=routes.float().abs()
            prob=mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)
            report.setdefault(f"layer_{_index}",{}).update(
                route_entropy_mean=(-(prob*prob.clamp_min(1e-12).log()).sum(-1)).mean().item(),
                route_max_mean=prob.max(-1).values.mean().item())
            attention=model.blocks[_index].attention
            if hasattr(attention,"query_weight"):
                ratios=[]
                query=attention.feature_map(inputs[0]) if attention.config.quadratic_features else inputs[0]
                for level in range(len(model.config.slots)):
                    qw=attention.query_weight[level]
                    if attention.config.shared_query_center:
                        qw=qw+attention.center[level].transpose(-1,-2)/attention.config.query_scale
                    direct=attention._slot_feature(attention.config.query_scale*torch.einsum(
                        "bthd,hdm->bthm",query,qw.to(query.dtype)))
                    rq=torch.einsum("bthm,hmn->bthn",routes[:,:,level],attention.route_weight[level].to(routes.dtype))
                    ratio=direct.float().norm(dim=-1)/rq.float().norm(dim=-1).clamp_min(1e-8)
                    ratios.append(ratio[labels!=-100].mean().item())
                report[f"layer_{_index}"]["direct_to_route_query_norm_ratio_before_balance"]=ratios
                if attention.config.balanced_bridge:
                    report[f"layer_{_index}"]["learned_route_branch_fraction"]=(.1+.8*attention.bridge_mix_raw.sigmoid()).detach().cpu().tolist()
            route_maps.append(prob[0].mean(-2).permute(1,0,2).cpu().tolist())
            return output,state
        block.attention.forward=capture

        def hook(module, inputs, output, index=index):
            q, k, v = inputs
            if getattr(module.config,"quadratic_features",False):k=module.feature_map(k)
            entry = report.get(f"layer_{index}",{})
            if getattr(module.config,"learn_read_scale",False):
                entry["learned_read_scale"]=module.read_log_scale.clamp(-1.38629436112,4.15888308336).exp().detach().cpu().tolist()
            if getattr(module.config,"quadratic_features",False):
                entry["quadratic_gain"]=(2*module.feature_map.gain_raw.sigmoid()).detach().cpu().tolist()
            if hasattr(module,"_gates"):
                gi, bi, go, bo = module._gates(k, 0,controls.get(index))
                entry.update(control_statistics_level=0,inner_beta_by_observed_token_type={},outer_beta_by_observed_token_type={},
                             inner_log_decay_by_observed_token_type={},outer_log_decay_by_observed_token_type={})
                for name, mask in masks.items():
                    entry["inner_beta_by_observed_token_type"][name] = bi[mask].float().mean().item() if mask.any() else None
                    entry["outer_beta_by_observed_token_type"][name] = bo[mask].float().mean().item() if mask.any() else None
                    entry["inner_log_decay_by_observed_token_type"][name] = gi[mask].float().mean().item() if mask.any() else None
                    entry["outer_log_decay_by_observed_token_type"][name] = go[mask].float().mean().item() if mask.any() else None
            state = output[1]
            ranks, dominant = [], []
            for relation in state.relations:
                singular = torch.linalg.svdvals(relation.memory.float())
                mass = singular/singular.sum(-1, keepdim=True).clamp_min(1e-12)
                ranks.append((-torch.sum(mass*mass.clamp_min(1e-12).log(), -1)).exp().mean().item())
                dominant.append(mass[..., 0].mean().item())
            entry["relation_effective_rank_mean"] = sum(ranks)/len(ranks) if ranks else None
            entry["relation_top_singular_mass_mean"] = sum(dominant)/len(dominant) if dominant else None
            entry["count_peak_fraction"] = [
                (s.count.max(-1).values/s.count.sum(-1).clamp_min(1e-12)).mean().item() for s in state.levels]
            report[f"layer_{index}"] = entry
            heatmaps.append([s.memory[0].float().mean(0).cpu().tolist() for s in state.relations])

        handles.append(block.attention.register_forward_hook(hook))
    def precision():
        return torch.autocast("cuda",dtype=torch.bfloat16) if args.device=="cuda" else nullcontext()
    with torch.no_grad(), precision():
        pred = model(ids).argmax(-1)
    for h in handles:
        h.remove()
    for module,original in originals:module.forward=original
    mask = labels != -100
    report["answer_token_accuracy_on_diagnostic_batch"] = ((pred == labels) & mask).sum().item()/mask.sum().item()
    report["protocol"]={"data_version":args.data_version,"task":args.task or "mixed","device":args.device,
                        "dtype":"bf16_autocast" if args.device=="cuda" else "float32_reference",
                        "seed":901111,"note":"token-type masks are analysis only; never passed to model"}
    if args.ablate_relation_read and model.config.memory_update=="legacy":
        from unittest.mock import patch
        if (len(set(model.config.slots))!=1 or model.config.chunk_size<=1 or
                model.config.execution not in ("packed","triton","staged")):
            raise ValueError("Read-only ablation requires the uniform packed/chunk execution path")
        def local_only(base,*unused,**kwargs):return base.softmax(-1)
        # Ignore all R messages at read time only. Keep every state write and
        # the query bypass unchanged, including the exact same input tokens.
        gpu_patch=patch("gpa.triton_route.dense_route",local_only) if args.device=="cuda" else nullcontext()
        with gpu_patch,patch("gpa.packed.dense_route_reference",local_only),torch.no_grad(),precision():
            no_r=model(ids).argmax(-1)
        report["no_relation_read_answer_token_accuracy"]=((no_r==labels)&mask).sum().item()/mask.sum().item()
        report["answer_predictions_changed_without_relation_read"]=((pred!=no_r)&mask).sum().item()/mask.sum().item()
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.plot and heatmaps:
        arrays={"relations":heatmaps[-1],"routes":route_maps[-1] if route_maps else None,
                "gates":gate_maps[-1] if gate_maps else None,"levels":len(model.config.slots),
                "protocol":report["protocol"]}
        path=Path(args.output).with_suffix(".arrays.json")
        path.write_text(json.dumps(arrays),encoding="utf-8")
        if importlib.util.find_spec("matplotlib") is None:
            print("matplotlib not installed; JSON diagnostics saved, optional plot skipped")
            return
        # Separate plotting from torch's runtime; arrays permit CPU-only redraws
        # and avoid duplicate OpenMP runtimes in some Windows installations.
        subprocess.run([sys.executable,str(Path(__file__).with_name("plot_diagnostics.py")),
                        str(path),"--output",args.output],check=True)


if __name__ == "__main__":
    main()
