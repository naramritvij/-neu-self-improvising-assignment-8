"""Persistent Modal runner. Nothing is launched merely by importing this file.

The default job prepares data on CPU. Every GPU job requires --acknowledge-compute.
Authenticate to the *course* workspace locally; never paste tokens into ChatGPT.
"""
from pathlib import Path
import os
import re
import subprocess
import modal

ROOT=Path(__file__).resolve().parent
app=modal.App('info7375-search-r1-ritvij-ghouse')
volume=modal.Volume.from_name('info7375-search-r1-team',create_if_missing=True)
# E5 flat index (64.6 GB), Wikipedia-2018 corpus and encoder, from modal_retrieval.py.
retrieval_volume=modal.Volume.from_name('info7375-ghouse-retrieval',version=2)
ASSETS='/retrieval/wiki18-e5'
image=(modal.Image.debian_slim(python_version='3.11')
       .apt_install('git')
       .pip_install('torch==2.6.0')
       .pip_install_from_requirements(str(ROOT/'requirements.txt'))
       .pip_install('faiss-cpu==1.10.0','fastapi==0.115.12','uvicorn==0.34.0')
       .env({'HF_HOME':'/vol/huggingface','PYTHONPATH':'/app','TOKENIZERS_PARALLELISM':'false','MODAL_VOLUME_NAME':'info7375-search-r1-team'})
       .add_local_dir(ROOT/'search_r1_team',remote_path='/app/search_r1_team',ignore=['__pycache__'])
       .add_local_dir(ROOT/'scripts',remote_path='/app/scripts',ignore=['__pycache__'])
       .add_local_dir(ROOT/'configs',remote_path='/app/configs'))


def call(arguments):
    subprocess.run(['python',*arguments],cwd='/app',check=True)


def safe_name(value):
    if value in {'.','..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+',value):
        raise ValueError('Use a simple run name, without slashes or spaces')
    return value

@app.function(image=image,cpu=4,memory=16384,timeout=3600,volumes={'/vol':volume})
def prepare():
    try:
        data=Path('/vol/data'); data.mkdir(parents=True,exist_ok=True)
        if not (data/'bamboogle_questions.jsonl').exists():
            call(['scripts/prepare_bamboogle.py','--output-dir',str(data)])
        if not (data/'sources.lock.json').exists():
            call(['scripts/prepare_training.py','--output-dir',str(data),
                  '--heldout-questions',str(data/'bamboogle_questions.jsonl')])
        return {'data_directory':str(data),'source_lock':(data/'sources.lock.json').read_text()}
    finally: volume.commit()

def start_local_retriever():
    """Serve E5 retrieval inside this container (co-located with the GPU job).

    Loads the 64.6 GB flat index into RAM, so the function needs ~96 GiB (MODAL_MEMORY=98304).
    Returns the URL and the provenance recorded in the asset manifest."""
    import json, time, urllib.request
    info=json.loads(Path(ASSETS,'assets_manifest.json').read_text())
    log=open('/tmp/retriever.log','w')
    server=subprocess.Popen(['python','scripts/serve_retrieval.py','--assets',ASSETS,'--host','127.0.0.1','--port','8000'],
                            cwd='/app',stdout=log,stderr=subprocess.STDOUT)
    start=time.monotonic()
    while True:
        if server.poll() is not None:
            raise RuntimeError('Retriever exited during startup: '+Path('/tmp/retriever.log').read_text()[-2000:])
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5) as r:
                health=json.loads(r.read())
            break
        except OSError:
            if time.monotonic()-start>3600: raise TimeoutError('Retriever did not become healthy within an hour')
            time.sleep(5)
    print(f"Retriever ready in {time.monotonic()-start:.0f}s: {health['documents']} documents",flush=True)
    corpus=f"{info['corpus']['id']}@{info['corpus']['revision'][:12]}+sha256:{info['corpus_sha256'][:12]}"
    retriever=f"{info['encoder']['id']}@{info['encoder']['revision'][:12]}+e5_Flat@{info['index']['revision'][:12]}"
    return 'http://127.0.0.1:8000/retrieve',corpus,retriever


@app.function(image=image,gpu=os.environ.get('MODAL_GPU','A100-80GB'),cpu=8,
              memory=int(os.environ.get('MODAL_MEMORY','32768')),timeout=86400,
              volumes={'/vol':volume,'/retrieval':retrieval_volume},
              secrets=([modal.Secret.from_name('search-r1-retrieval')] if os.environ.get('USE_RETRIEVAL_SECRET')=='1' else []))
def gpu_run(job: str,algorithm: str,profile: str,run_name: str,method: str,
            retrieval_url: str,corpus_id: str,retriever_id: str,checkpoint: str,steps: int,limit: int,resume: bool,minutes: int=0):
    out=Path('/vol/runs')/safe_name(run_name)
    if retrieval_url=='local':
        retrieval_url,corpus_id,retriever_id=start_local_retriever()
    retrieval=['--retrieval-url',retrieval_url,'--corpus-id',corpus_id,'--retriever-id',retriever_id]
    t0=__import__('time').monotonic()
    try:
        if job=='train':
            if algorithm not in ('grpo','ppo') or profile not in ('smoke','short','experiment'): raise ValueError('Invalid training mode')
            if not all((retrieval_url,corpus_id,retriever_id)): raise ValueError('Configure a reachable retrieval service first')
            config=f'configs/{algorithm}'+{'smoke':'_smoke','short':'_short'}.get(profile,'')+'.json'
            cmd=['scripts/train.py','--config',config,'--data-dir','/vol/data','--output',str(out),*retrieval]
            if steps: cmd+=['--steps',str(steps)]
            if minutes: cmd+=['--max-minutes',str(minutes)]
            if resume:
                if not checkpoint: raise ValueError('Resume requires the latest complete checkpoint path')
                cmd+=['--resume',checkpoint]
            call(cmd)
        elif job=='evaluate':
            if method not in ('direct','base_search','own_grpo','own_ppo','released_grpo','released_ppo'): raise ValueError('Invalid evaluation method')
            cmd=['scripts/infer.py','--method',method,'--data-dir','/vol/data','--output',str(out)]
            if method!='direct':
                if not all((retrieval_url,corpus_id,retriever_id)): raise ValueError('Search inference requires retrieval')
                cmd+=retrieval
            if checkpoint: cmd+=['--checkpoint',checkpoint]
            if limit: cmd+=['--limit',str(limit)]
            if resume: cmd+=['--resume']
            call(cmd)
            # The complete benchmark scorer rejects partial diagnostics by default.
            score=['scripts/evaluate.py','--gold','/vol/data/bamboogle_labels.jsonl',
                  '--predictions',str(out/'predictions.jsonl'),'--method','direct' if method=='direct' else ('ppo' if method.endswith('ppo') else 'grpo'),
                  '--output',str(out/'metrics.json')]
            if limit: score+=['--allow-partial']
            if resume and (out/'metrics.json').exists(): score+=['--overwrite']
            call(score)
        else: raise ValueError('Unknown GPU job')
        return {'output_directory':str(out),'files':[str(p.relative_to(out)) for p in out.glob('*.json')],
                'container_seconds':round(__import__('time').monotonic()-t0)}
    finally: volume.commit()

@app.local_entrypoint()
def main(job: str='prepare',algorithm: str='grpo',profile: str='smoke',run_name: str='smoke-grpo',
         method: str='direct',retrieval_url: str='',corpus_id: str='',retriever_id: str='',
         checkpoint: str='',steps: int=0,limit: int=0,resume: bool=False,minutes: int=0,acknowledge_compute: bool=False):
    if steps<0 or limit<0: raise ValueError('Steps and limit cannot be negative')
    if job=='prepare':
        print(prepare.remote()); return
    if not acknowledge_compute:
        raise ValueError('GPU jobs can spend workspace credits. Verify the course workspace and add --acknowledge-compute.')
    print(gpu_run.remote(job,algorithm,profile,run_name,method,retrieval_url,corpus_id,retriever_id,checkpoint,steps,limit,resume,minutes))
