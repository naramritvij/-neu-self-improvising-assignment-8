"""Read-only checks; no downloads, repository changes or GPU jobs."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    result={'python':sys.version.split()[0],'git_installed':bool(shutil.which('git')),
      'gh_installed':bool(shutil.which('gh')),'modal_installed':bool(importlib.util.find_spec('modal')),
      'data_prepared':(ROOT/'data/processed/sources.lock.json').exists(),
      'starter_fetched':(ROOT/'vendor/tiny-zero-lora/train.py').exists()}
    if importlib.util.find_spec('torch'):
        import torch
        result.update(torch=torch.__version__,cuda_available=torch.cuda.is_available())
    result['note']='A locally available GPU is unnecessary when using Modal; authenticated workspace access is still required.'
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
