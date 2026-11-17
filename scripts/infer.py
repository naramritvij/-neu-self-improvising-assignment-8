"""Response-only predictions and separate traces; no gold labels reach the model."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from search_r1_team.config import Config
from search_r1_team.data import read_jsonl
from search_r1_team.model import HFPolicy
from search_r1_team.retrieval import HTTPRetriever
from search_r1_team.rollout import run_episodes
from search_r1_team.runtime import environment,seed_all,sha256,write_json

METHODS=['direct','base_search','own_grpo','own_ppo','released_grpo','released_ppo']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/evaluation.json')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/processed')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,help='Own training checkpoint directory containing actor/')
    p.add_argument('--retrieval-url'); p.add_argument('--corpus-id'); p.add_argument('--retriever-id')
    p.add_argument('--limit',type=int,help='Diagnostic only; never reported as the full benchmark')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--batch-size',type=int,default=25,help='questions generated together per batch')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='cpu is for dry runs with a small model')
    a=p.parse_args()
    if a.limit is not None and a.limit<1: p.error('Diagnostic limit must be positive')
    own=a.method.startswith('own_'); search=a.method!='direct'
    if own and not a.checkpoint: p.error('An own_* run requires your trained checkpoint, not a released model')
    if not own and a.checkpoint: p.error('Only own_* runs accept an adapter')
    if search and not all((a.retrieval_url,a.corpus_id,a.retriever_id)): p.error('Search inference needs retrieval URL and provenance')
    if a.output.exists() and not a.resume: p.error('Output exists; use a new folder or --resume')
    cfg=Config.load(a.config)
    # base_search: the untrained instruct model with the Search-R1 prompt and retriever.
    source_key='direct' if own or a.method=='base_search' else a.method
    lock=json.loads((a.data_dir/'sources.lock.json').read_text())['models'][source_key]
    cfg.model_id,cfg.model_revision=lock['id'],lock['revision']
    if search: cfg.algorithm=a.method.split('_')[-1] if a.method!='base_search' else 'grpo'
    training_metadata=None
    if own:
        metadata=json.loads((a.checkpoint/'model_metadata.json').read_text())
        if metadata['base_model']!=cfg.model_id or metadata['base_revision']!=cfg.model_revision:
            p.error('Adapter base-model revision does not match the frozen model')
        training_metadata=json.loads((a.checkpoint/'checkpoint_metadata.json').read_text())
        # Algorithm is read from a sidecar, not by unpickling a checkpoint.
        ckpt_config=json.loads((a.checkpoint/'training_config.json').read_text())
        if ckpt_config['algorithm']!=a.method.removeprefix('own_'): p.error('Checkpoint belongs to the other algorithm')
    questions=read_jsonl(a.data_dir/'bamboogle_questions.jsonl')
    selected=questions[:a.limit] if a.limit else questions
    retrieval=HTTPRetriever(a.retrieval_url,corpus_id=a.corpus_id,retriever_id=a.retriever_id) if search else None
    if retrieval: retrieval.search('Wikipedia',1)
    signature={'method':a.method,'config':cfg.to_dict(),'questions_sha256':sha256(a.data_dir/'bamboogle_questions.jsonl'),
       'checkpoint':str(a.checkpoint.resolve()) if own else None,
       'own_training':training_metadata,
       'adapter_sha256':sha256(a.checkpoint/'actor'/'adapter_model.safetensors') if own else None,
       'retrieval':retrieval.provenance if retrieval else None,'limit':a.limit}
    done=set()
    if a.resume:
        prior=json.loads((a.output/'manifest.json').read_text())
        if prior['signature']!=signature: p.error('Resume configuration differs from the saved run')
        pred_path=a.output/'predictions.jsonl'
        previous=read_jsonl(pred_path) if pred_path.exists() and pred_path.stat().st_size else []
        done={x['id'] for x in previous}
        if not done.issubset({x['id'] for x in selected}): p.error('Existing predictions contain unexpected IDs')
    seed_all(cfg.seed)
    policy=HFPolicy(cfg,adapter_path=str(a.checkpoint/'actor') if own else None,training=False,device=a.device)
    a.output.mkdir(parents=True,exist_ok=True)
    manifest={'status':'running','signature':signature,'environment':environment(),
        'total_questions':len(questions),'selected_questions':len(selected),
        'coverage':'full' if len(selected)==len(questions) else 'partial_diagnostic',
        'checkpoint_origin':'own_lora_training' if own else ('base_instruct_with_search' if a.method=='base_search' else 'released_full_checkpoint' if search else 'base_instruct'),
        'prompt_parity':'documented_compact_prompt_not_exact_flashrag_baseline'}
    write_json(a.output/'manifest.json',manifest)
    try:
        todo=[row for row in selected if row['id'] not in done]
        for i in range(0,len(todo),a.batch_size):
            chunk=todo[i:i+a.batch_size]; start=time.monotonic()
            trs=run_episodes(policy,[row['question'] for row in chunk],cfg,retriever=retrieval,sample=False)
            seconds=(time.monotonic()-start)/len(chunk)
            for row,tr in zip(chunk,trs):
                # Persist trace first, then prediction. An interrupted tail can be rerun deterministically.
                write_json(a.output/'traces'/f'{row["id"]}.json',{
                    'id':row['id'],'question':row['question'],'events':tr.events,
                    'stop_reason':tr.stop_reason,'action_tokens':len(tr.action_positions)})
                with (a.output/'predictions.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'id':row['id'],'prediction':tr.response,
                        'stop_reason':tr.stop_reason,'seconds':seconds},ensure_ascii=False)+'\n'); f.flush()
                done.add(row['id'])
            print(f'{a.method}: {len(done)}/{len(selected)}',flush=True)
        manifest['status']='completed'
    except Exception as exc:
        manifest['status']='failed'; manifest['error_type']=type(exc).__name__
        write_json(a.output/'manifest.json',manifest); raise
    write_json(a.output/'manifest.json',manifest)

if __name__=='__main__': main()
