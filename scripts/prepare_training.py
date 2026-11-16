"""Freeze a disjoint NQ/HotpotQA train/validation subset and source revisions."""
import argparse
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from search_r1_team.data import read_jsonl,write_jsonl
from search_r1_team.runtime import sha256,write_json

MODELS={
 'direct':'Qwen/Qwen2.5-3B-Instruct',
 'released_grpo':'PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.2',
 'released_ppo':'PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.2',
}
DATASET='PeterJinGo/nq_hotpotqa_train'

def key(question): return ' '.join(question.casefold().split())

def split_records(rows,heldout,train_size,val_size,seed):
    excluded={key(x['question']) for x in heldout}; seen=set(); clean=[]
    for row in rows:
        question=row.get('question'); answers=row.get('golden_answers',row.get('answers'))
        if not isinstance(question,str) or not question.strip(): continue
        if not isinstance(answers,list) or not answers or any(not isinstance(x,str) or not x.strip() for x in answers): continue
        k=key(question)
        if k in excluded or k in seen: continue
        seen.add(k)
        clean.append({'id':str(row.get('data_source','qa'))+'_'+str(row['id']),
                      'question':question,'answers':answers,'source':row.get('data_source','unknown')})
    random.Random(seed).shuffle(clean)
    if len(clean)<train_size+val_size: raise ValueError('Not enough clean examples for the requested disjoint splits')
    return clean[val_size:val_size+train_size],clean[:val_size]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/processed')
    p.add_argument('--heldout-questions',type=Path,default=ROOT/'data/processed/bamboogle_questions.jsonl')
    p.add_argument('--train-size',type=int,default=4000); p.add_argument('--val-size',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if min(a.train_size,a.val_size)<1: p.error('Both split sizes must be positive')
    for name in ('train.jsonl','validation.jsonl','sources.lock.json','training_manifest.json'):
        if (a.output_dir/name).exists(): p.error('Prepared training data already exists; use a fresh output folder')
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api=HfApi()
    sources={'dataset':{'id':DATASET,'revision':api.dataset_info(DATASET).sha},'models':{}}
    for name,model_id in MODELS.items():
        sources['models'][name]={'id':model_id,'revision':api.model_info(model_id).sha}
    ds=load_dataset(DATASET,revision=sources['dataset']['revision'],split='train')
    train,val=split_records(ds,read_jsonl(a.heldout_questions),a.train_size,a.val_size,a.seed)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    write_jsonl(a.output_dir/'train.jsonl',train); write_jsonl(a.output_dir/'validation.jsonl',val)
    write_json(a.output_dir/'sources.lock.json',sources)
    write_json(a.output_dir/'training_manifest.json',{
        'source':sources['dataset'],'source_split':'train','seed':a.seed,
        'training_examples':len(train),'validation_examples':len(val),'deduplicated':True,
        'bamboogle_used_for_training':False,'heldout_questions_sha256':sha256(a.heldout_questions),
        'train_sha256':sha256(a.output_dir/'train.jsonl'),
        'validation_sha256':sha256(a.output_dir/'validation.jsonl')})
    print('Frozen training/validation data and revisions:',a.output_dir)

if __name__=='__main__': main()
