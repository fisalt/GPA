"""Public reproducibility metadata: versions and source hashes, no private paths."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import torch
import gpa


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(gpa.__file__).parent
    data={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
          'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'gpa':gpa.__version__,
          'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
              ('attention.py','ops.py','packed.py','staged.py','features.py','retrieval.py','triton_route.py','training_graph.py')}}
    for name in ('triton','flash-linear-attention','pytest'):
        try:data[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:data[name]=None
    try:
        module=importlib.import_module('fla.ops.gated_delta_rule.chunk')
        data['fla_public_chunk_sha256']=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    except ImportError:data['fla_public_chunk_sha256']=None
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2),flush=True)


if __name__=='__main__':main()
