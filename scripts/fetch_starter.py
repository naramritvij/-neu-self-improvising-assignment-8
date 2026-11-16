"""Fetch the inspected upstream snapshot. No training is launched."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/suhabe/tiny-zero-lora.git'
COMMIT = '3659acfad61dab2f90fa46294566ff6808452e87'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, default=ROOT/'vendor'/'tiny-zero-lora')
    args = p.parse_args()
    if not shutil.which('git'): p.error('Git must be installed and on PATH.')
    if args.destination.exists(): p.error('Destination exists; refusing to overwrite it.')
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(['git','clone','--no-checkout',URL,str(args.destination)], check=True, timeout=300)
        subprocess.run(['git','-C',str(args.destination),'checkout','--detach',COMMIT], check=True, timeout=60)
        actual = subprocess.check_output(['git','-C',str(args.destination),'rev-parse','HEAD'], text=True, timeout=30).strip()
        if actual != COMMIT: raise RuntimeError('Pinned commit verification failed.')
    except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
        p.error(f'Fetch stopped: {exc}. Any partial folder was preserved for inspection.')
    print(f'Pinned starter: {args.destination}\nCommit: {actual}')
    print('Read the source and docs/MODAL_SETUP.md before installing GPU packages or training.')

if __name__ == '__main__': main()
