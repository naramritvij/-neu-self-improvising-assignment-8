"""Train compact Search-R1 with LoRA/GRPO or LoRA/PPO; requires a real retrieval API."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from search_r1_team.algorithms import prepare_batch,update_batch
from search_r1_team.config import Config
from search_r1_team.data import read_jsonl
from search_r1_team.metrics import exact_match,extract_answer
from search_r1_team.model import HFPolicy
from search_r1_team.retrieval import HTTPRetriever
from search_r1_team.rollout import run_episodes
from search_r1_team.runtime import environment,seed_all,sha256,write_json,save_state,restore_state,commit_mounted_volume
from scripts.prepare_training import key


def assert_splits(train,val,heldout):
    sets=[{key(x['question']) for x in rows} for rows in (train,val,heldout)]
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:
        raise ValueError('Training, validation and Bamboogle questions must be disjoint')
    for row in train+val: exact_match('',row['answers'])


def train(args):
    cfg=Config.load(args.config)
    if args.steps is not None: cfg.steps=args.steps
    lock=json.loads((args.data_dir/'sources.lock.json').read_text())['models']['direct']
    if cfg.model_id!=lock['id']: raise ValueError('Config model differs from frozen base model')
    cfg.model_revision=lock['revision']; cfg.validate()
    seed_all(cfg.seed)
    rows=read_jsonl(args.data_dir/'train.jsonl'); val=read_jsonl(args.data_dir/'validation.jsonl')
    heldout=read_jsonl(args.data_dir/'bamboogle_questions.jsonl'); assert_splits(rows,val,heldout)
    retrieval=HTTPRetriever(args.retrieval_url,corpus_id=args.corpus_id,retriever_id=args.retriever_id)
    # Verify the remote service before loading a GPU model. This is not an evaluation question.
    retrieval.search('Wikipedia',1)
    out=args.output
    if out.exists() and not args.resume: raise ValueError('Run folder exists; choose a new name or resume the latest checkpoint')
    provenance={'train_sha256':sha256(args.data_dir/'train.jsonl'),
        'validation_sha256':sha256(args.data_dir/'validation.jsonl'),
        'bamboogle_questions_sha256':sha256(args.data_dir/'bamboogle_questions.jsonl'),
        'retrieval':retrieval.provenance,'base_revision':cfg.model_revision}
    if args.resume:
        old=json.loads((out/'manifest.json').read_text())
        if old['provenance']!=provenance: raise ValueError('Resume data, retrieval or model revision changed')
    out.mkdir(parents=True,exist_ok=True)
    policy=HFPolicy(cfg,device=args.device)
    actor_params=policy.actor_parameters()
    if not actor_params: raise ValueError('No LoRA actor parameters found; refusing full-model training')
    actor=torch.optim.AdamW(actor_params,lr=cfg.learning_rate)
    critic=torch.optim.AdamW(policy.critic_parameters(),lr=cfg.critic_learning_rate) if cfg.algorithm=='ppo' else None
    start=0
    if args.resume:
        policy.load_training(args.resume); start=restore_state(args.resume,actor,critic,cfg)
        existing=(out/'training.jsonl').read_text().splitlines() if (out/'training.jsonl').exists() else []
        if existing and json.loads(existing[-1])['step']!=start:
            raise ValueError('Resume must match the latest logged step; use a clean copy to recover an older checkpoint')
    manifest={'status':'running','implementation':'compact_single_gpu_lora_not_upstream_verl',
        'environment':environment(),'config':cfg.to_dict(),'provenance':provenance,
        'actor_trainable_parameters':sum(p.numel() for p in actor_params),
        'critic_trainable_parameters':sum(p.numel() for p in policy.critic_parameters()),
        'base_model_parameters_frozen':True,'bamboogle_not_used_for_selection':True,
        'run_kind':'smoke' if cfg.steps<=2 else 'experiment'}
    write_json(out/'manifest.json',manifest)
    started=time.monotonic()
    try:
        for step in range(start+1,cfg.steps+1):
            t0=time.monotonic(); trajectories=[]; rewards=[]; groups=[]
            # All prompts x group samples are generated together (one batched call per turn).
            # Group members stay consecutive, as group_advantages expects.
            picked=[row for row in random.sample(rows,min(cfg.prompts_per_step,len(rows))) for _ in range(cfg.group_size)]
            trajectories=run_episodes(policy,[row['question'] for row in picked],cfg,retriever=retrieval,sample=True)
            rewards=[exact_match(extract_answer(tr.response,tagged=True),row['answers']) for tr,row in zip(trajectories,picked)]
            groups=[row['id'] for row in picked]
            t_rollout=time.monotonic()-t0
            batch=prepare_batch(policy,trajectories,rewards,cfg)
            warmup=cfg.algorithm=='ppo' and step<=cfg.critic_warmup_steps
            stats=update_batch(policy,batch,cfg,actor,critic,actor_update=not warmup)
            stats.update({'step':step,'reward_mean':sum(rewards)/len(rewards),
              'rollouts':len(rewards),'prompts':len(set(groups)),'actor_updated':not warmup,
              'generated_tokens':sum(len(x.action_positions) for x in trajectories),
              'mean_searches':sum(sum(e['kind']=='retrieval' for e in x.events) for x in trajectories)/len(trajectories),
              'seconds':time.monotonic()-t0,'rollout_seconds':t_rollout,
              'elapsed_seconds_this_session':time.monotonic()-started})
            # Save a small real trace sample, never all training data/checkpoints in Git.
            if step==1 or step%cfg.save_every==0:
                write_json(out/'traces'/f'step_{step:05d}.json',[
                    {'id':i,'reward':r,'response':x.response,'events':x.events,'stop_reason':x.stop_reason}
                    for i,r,x in list(zip(groups,rewards,trajectories))[:2]])
            if cfg.eval_every and step%cfg.eval_every==0:
                subset=val[:cfg.validation_limit]
                trs=run_episodes(policy,[row['question'] for row in subset],cfg,retriever=retrieval,sample=False)
                scores=[exact_match(extract_answer(tr.response,tagged=True),row['answers']) for tr,row in zip(trs,subset)]
                stats['validation_em']=sum(scores)/len(scores)
                stats['validation_n']=len(scores)
            with (out/'training.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(stats)+'\n'); handle.flush()
            print(json.dumps(stats),flush=True)
            out_of_time=bool(args.max_minutes) and (time.monotonic()-started)/60>=args.max_minutes
            if step%cfg.save_every==0 or step==cfg.steps or out_of_time:
                ckpt=out/'checkpoints'/f'step_{step:05d}'
                policy.save(ckpt); write_json(ckpt/'training_config.json',cfg.to_dict())
                write_json(ckpt/'checkpoint_metadata.json',{'step':step,'algorithm':cfg.algorithm,'run_kind':manifest['run_kind']})
                save_state(ckpt,step=step,actor=actor,critic=critic,config=cfg)
                write_json(out/'latest_checkpoint.json',{'step':step,'path':str(ckpt),'actor':str(ckpt/'actor')})
                commit_mounted_volume()
            if out_of_time:
                manifest['stopped_by_time_budget_minutes']=args.max_minutes
                break
        manifest['status']='completed'; manifest['completed_steps']=step
    except Exception as exc:
        manifest['status']='failed'; manifest['error_type']=type(exc).__name__
        write_json(out/'manifest.json',manifest)
        raise
    write_json(out/'manifest.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--data-dir',type=Path,default=ROOT/'data/processed')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--resume',type=Path)
    p.add_argument('--steps',type=int); p.add_argument('--retrieval-url',required=True)
    p.add_argument('--corpus-id',required=True); p.add_argument('--retriever-id',required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='cpu is for dry runs with a small model')
    p.add_argument('--max-minutes',type=float,default=0,help='stop (and checkpoint) after the step that exceeds this wall-clock budget')
    a=p.parse_args(); train(a)

if __name__=='__main__': main()
