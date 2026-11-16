"""Run evidence and RNG state utilities; never log credentials."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import torch


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def environment():
    versions={}
    for name in ['torch','transformers','peft','datasets','huggingface-hub','modal','numpy']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError): commit=None
    return {'timestamp_utc':datetime.now(timezone.utc).isoformat(),'versions':versions,'git_commit':commit,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); temp.replace(path)


def save_state(path,*,step,actor,critic,config):
    state={'step':step,'actor_optimizer':actor.state_dict(),
           'critic_optimizer':critic.state_dict() if critic else None,'config':config.to_dict(),
           'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
           'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(state,Path(path)/'training_state.pt')


def restore_state(path,actor,critic,config):
    # Only load a checkpoint that YOU generated. torch.load(weights_only=False)
    # restores Python RNG state and must never consume an untrusted checkpoint.
    state=torch.load(Path(path)/'training_state.pt',map_location='cpu',weights_only=False)
    old=dict(state['config']); new=config.to_dict()
    # Extending the step count is allowed; other changes invalidate exact resume.
    old.pop('steps'); new.pop('steps')
    if old!=new: raise ValueError('Resume configuration changed beyond the total step budget')
    actor.load_state_dict(state['actor_optimizer'])
    if critic: critic.load_state_dict(state['critic_optimizer'])
    random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
    return int(state['step'])


def commit_mounted_volume():
    """Explicit checkpoint commit when running inside the configured Modal volume."""
    import os
    name=os.environ.get("MODAL_VOLUME_NAME")
    if name:
        import modal
        modal.Volume.from_name(name).commit()
