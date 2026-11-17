"""Separate class-starter image. First run python scripts/fetch_starter.py locally."""
from pathlib import Path
import os
import subprocess
import modal
ROOT=Path(__file__).resolve().parent
VENDOR=ROOT/'vendor/tiny-zero-lora'
# Only check locally: inside the container this file is re-imported from /root, where the vendor
# folder is not at this relative path (the image places it at /app/vendor/tiny-zero-lora).
if modal.is_local() and not (VENDOR/'requirements.txt').exists():
    raise RuntimeError('Fetch the pinned class starter before importing this Modal app')
app=modal.App('info7375-tiny-zero-starter-team')
volume=modal.Volume.from_name('info7375-search-r1-team',create_if_missing=True)
image=(modal.Image.from_registry('nvidia/cuda:13.1.1-devel-ubuntu24.04',add_python='3.12')
    .apt_install('git','libnuma1').pip_install_from_requirements(str(VENDOR/'requirements.txt'))
    .env({'HF_HOME':'/vol/huggingface'})
    .add_local_dir(VENDOR,'/app/vendor/tiny-zero-lora',ignore=['.venv','**/__pycache__'])
    .add_local_file(ROOT/'scripts/run_starter.py','/app/scripts/run_starter.py'))

@app.function(image=image,gpu=os.environ.get('MODAL_STARTER_GPU','B300'),cpu=8,memory=32768,
              timeout=86400,volumes={'/vol':volume})
def run(smoke: bool,steps: int,run_name: str,reduced: bool=False):
    import re
    if run_name in {'.','..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+',run_name): raise ValueError('Invalid run name')
    cmd=['python','/app/scripts/run_starter.py','--output',f'/vol/starter/{run_name}','--steps',str(steps)]
    if smoke: cmd+=['--smoke']
    if reduced: cmd+=['--reduced']
    try: subprocess.run(cmd,check=True)
    finally: volume.commit()

@app.local_entrypoint()
def main(run_name: str='starter-smoke',steps: int=2000,full: bool=False,reduced: bool=False,acknowledge_compute: bool=False):
    if not acknowledge_compute: raise ValueError('Check your workspace/budget, then add --acknowledge-compute')
    run.remote(not full,steps,run_name,reduced)
