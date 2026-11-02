"""One-process integration check; CUDA initialization is shared across phases."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import torch
import pytest
import transformers


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cuda',action='store_true')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cpu=pytest.main(['-q','tests/test_hf_qwen.py','tests/test_finetune_data.py','-k','not cuda','--tb=short'])
    if cpu:raise SystemExit(int(cpu))
    previous=sys.argv
    try:
        sys.argv=['smoke_qwen_conversion.py','--output',str(out/'pipeline')]
        runpy.run_path(str(Path(__file__).with_name('smoke_qwen_conversion.py')),run_name='__main__')
    finally:sys.argv=previous
    gpu=None
    if args.cuda:
        if not torch.cuda.is_available():raise RuntimeError('CUDA was requested but is unavailable')
        gpu=pytest.main(['-q','tests/test_hf_qwen.py','-k','cuda','--tb=short'])
        if gpu:raise SystemExit(int(gpu))
    record=dict(status='PASS',torch=torch.__version__,transformers=transformers.__version__,
                cpu_tests_passed=True,cli_conversion_training_resume_merge=True,
                cuda_test_passed=bool(args.cuda),real_fineweb_quality_claim=False)
    (out/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
