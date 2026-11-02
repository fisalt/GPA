"""Scale the validation-selected architecture; never transfer trained weights."""
import json
import subprocess
import sys
from pathlib import Path


def main():
    candidates={}
    for name in ("paired_product_v3","paired_balanced_v3","paired_context_v3","paired_marginal_v3","paired_quadratic_v3"):
        path=Path("..")/name/"results.json"
        if path.is_file():candidates[name]=json.loads(path.read_text())
    def score(run):
        t=run["evaluations"]["256"]["tasks"]
        return min(x["free_generation_exact_match" if k=="uuid4" else "query_exact_match_teacher_forced"] for k,x in t.items())
    name=max(candidates,key=lambda k:score(candidates[k]))
    if score(candidates[name])>=.95:
        print("All-task validation target reached; skip extra scale trial",flush=True);return
    cfg=candidates[name]["config"]
    features=["--"+key.replace("_","-") for key in ("balanced_bridge","context_controls","relation_marginal","quadratic_features") if cfg.get(key)]
    selection={"architecture_selected_from":name,"validation_worst_task":score(candidates[name]),
               "features":features,"initialization":"fresh random weights, no checkpoint transfer"}
    Path("../scaled_selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    print(selection,flush=True)
    common=["--width","256","--layers","4","--data-version","3","--sequence-balanced","--steps","20000",
            "--easy-curriculum","--curriculum-span","1000","--cosine","--warmup-steps","100",
            "--min-lr",".0002","--eval-samples","128"]
    script=[sys.executable,"scripts/train_retrieval.py"]
    subprocess.run(script+["--rsa-update","--aligned-query","--orthogonal-centers","--slots","32,32",
                   "--paper-product-read","--cuda-graph",*features,*common,"--output","../paired_scaled_v3"],check=True)
    subprocess.run(script+["--attention","transformer","--transformer-rope",*common,
                          "--output","../control_scaled_v3"],check=True)


if __name__=="__main__":main()
