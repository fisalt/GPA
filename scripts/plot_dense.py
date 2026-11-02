"""Static scientific comparison; label validation/extrapolation explicitly."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--summary",default="docs/results/v3/summary.json")
    p.add_argument("--output",default="docs/results/v3/retrieval.png")
    args=p.parse_args()
    runs=json.loads(Path(args.summary).read_text())
    names={"paired_product_v3":"GPA: product", "paired_context_v3":"GPA: +context",
           "paired_quadratic_v3":"GPA: +quadratic", "paired_temporal_v3":"GPA: +QK init",
           "transformer_rope_v3":"Transformer (RoPE)"}
    selected=[(name,label) for name,label in names.items() if name in runs]
    tasks=("single","uuid4","mqar","overwrite")
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),sharey=True)
    for ax,length in zip(axes,("256","512")):
        w=.8/len(selected)
        for i,(name,label) in enumerate(selected):
            if length not in runs[name]["evaluations"]:continue
            t=runs[name]["evaluations"][length]["tasks"]
            values=[100*t[k]["free_generation_exact_match" if k=="uuid4" else "query_exact_match_teacher_forced"] for k in tasks]
            ax.bar(np.arange(4)+(i-(len(selected)-1)/2)*w,values,w,label=label)
        ax.set_xticks(range(4),["Single","4-token ID","MQAR","Overwrite"])
        ax.set_ylim(0,105);ax.grid(axis="y",alpha=.25);ax.set_axisbelow(True)
        ax.set_title("T256: trained length" if length=="256" else "T512: extrapolation")
    axes[0].set_ylabel("Exact match (%)")
    axes[1].legend(loc="lower right",fontsize=8)
    fig.suptitle("Synthetic V3 validation (seed 900000); 20k steps, pure answer CE\n4-token ID: free generation; MQAR: per-query teacher-forced EM; not official RULER")
    fig.tight_layout()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=170)
    print(out)


if __name__=="__main__":main()
