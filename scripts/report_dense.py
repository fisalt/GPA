"""Collect completed runs without mixing data versions, budgets or held-outs."""
import argparse
import json
from pathlib import Path


RUNS = {
    "paper_run":"counted4", "paper_2x32":"counted2", "paper_raw":"counted_raw",
    "paper_deep":"counted_deep", "control_deep":"transformer_deep",
    "control_run":"transformer", "paper_transport":"counted_transport",
    "paired_gdn":"paired_gdn4", "paper_fullcontrol":"counted_full_controls",
    "control_slow":"transformer_slow", "paper_slot":"counted_slot_decay",
    "paired_2x32":"paired_gdn2_orthogonal", "paper_final":"counted_strong_v2",
    "control_final":"transformer_v2", "heldout_paper":"counted_strong_v2_holdout",
    "heldout_control":"transformer_v2_holdout",
    "paired_product_v3":"paired_product_v3", "control_rope_v3":"transformer_rope_v3",
    "paired_positive_v3":"paired_positive_v3",
    "paired_calibrated_v3":"paired_calibrated_v3",
    "paired_balanced_v3":"paired_balanced_v3",
    "heldout_balanced_v3":"paired_balanced_v3_holdout",
    "paired_multirecord_v3":"paired_multirecord_v3",
    "paired_context_v3":"paired_context_v3",
    "paired_marginal_v3":"paired_marginal_v3",
    "paired_quadratic_v3":"paired_quadratic_v3",
    "paired_readgate_v3":"paired_readgate_v3",
    "paired_temporal_v3":"paired_temporal_v3",
    "paired_widekey_v3":"paired_widekey_v3","control_widekey_v3":"transformer_widekey_v3",
    "paired_dense4_v3":"paired_dense4_v3",
    "paired_temperature_v3b":"paired_temperature_v3b","control_temporal_v3b":"transformer_temporal_v3b",
    "paired_memory64_v3b":"paired_memory64_v3b","control_param_v3b":"transformer_param_v3b",
    "heldout_fast_v3":"selected_gpa_fast_v3_holdout",
    "control_multirecord_v3":"transformer_multirecord_v3",
    "heldout_multirecord_v3":"paired_multirecord_v3_holdout",
    "paired_scaled_v3":"paired_scaled_v3", "control_scaled_v3":"transformer_scaled_v3",
    "heldout_best_v3":"selected_gpa_v3_holdout", "heldout_best_control_v3":"matched_transformer_v3_holdout",
    "heldout_calibrated_v3":"paired_calibrated_v3_holdout",
    "heldout_positive_v3":"paired_positive_v3_holdout",
    "heldout_product_v3":"paired_product_v3_holdout",
    "heldout_rope_v3":"transformer_rope_v3_holdout",
}


def score(tasks, key):
    task=tasks[key]
    return 100*task["free_generation_exact_match" if key=="uuid4" else "query_exact_match_teacher_forced"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="runs")
    p.add_argument("--output",default="docs/results/v3/summary.json")
    args=p.parse_args()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    runs={}
    for directory,name in RUNS.items():
        path=Path(args.root)/directory/"results.json"
        if path.is_file():runs[name]=json.loads(path.read_text())
    out.write_text(json.dumps(runs,indent=2)+"\n",encoding="utf-8")
    text=["# Dense GPA retrieval experiments", "",
          "Synthetic structured retrieval, not official RULER/NIAH or BABILong. All runs start from random weights.",
          "UUID4 uses actual free-generation exact match; MQAR uses per-query teacher-forced EM (three queries/example).",
          "V1 overwrite usually queries an unchanged key. V2 always queries the single last overwrite and admits a recency shortcut.",
          "V3 overwrites four different keys and randomly queries one of them; use V3 for the main comparison.",
          "T256 is the training length; T512 is extrapolation. Do not pool versions, curricula or test seeds.", ""]
    for version in (1,2,3):
        group={k:v for k,v in runs.items() if v["training_metadata"].get("data_version",1)==version}
        if not group:continue
        text += [f"## Data version {version}", "",
                 "| Model | Steps | Curriculum span | Loss average | Seed | T | Single | UUID4 | MQAR | Overwrite |",
                 "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        for name,run in group.items():
            meta=run["training_metadata"]
            for length,result in run["evaluations"].items():
                scores=[score(result["tasks"],k) for k in ("single","uuid4","mqar","overwrite")]
                curriculum=str(meta.get("curriculum_span",200)) if meta.get("easy_curriculum") else "multi-record from start"
                row=[name,str(run["steps"]),curriculum,
                     meta.get("loss_balance","mean_over_answer_tokens"),str(meta["test_seed"]),length,
                     *(f"{s:.2f}%" for s in scores)]
                text.append("| "+" | ".join(row)+" |")
        text.append("")
    out.with_suffix(".md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(json.dumps({"completed_runs":len(runs),"output":str(out)}))


if __name__=="__main__":main()
