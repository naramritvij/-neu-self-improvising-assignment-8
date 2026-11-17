"""E5 + FAISS dense retrieval API; defaults to localhost, not an exposed public service."""
import argparse
import hmac
import json
import os
from pathlib import Path
import sys
import threading
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from search_r1_team.corpus import IndexedCorpus


def create_app(assets: Path,device: str='cpu'):
    import faiss
    import numpy as np
    import torch
    from fastapi import FastAPI,HTTPException,Request
    from transformers import AutoTokenizer,AutoModel
    info=json.loads((assets/'assets_manifest.json').read_text())
    corpus=IndexedCorpus(assets/'wiki-18.jsonl')
    index=faiss.read_index(str(assets/'e5_Flat.index'))
    if index.ntotal!=len(corpus): raise ValueError('Index/corpus row counts differ; cannot safely map retrieved passages')
    encoder=info['encoder']
    # Prefer the encoder snapshot saved next to the index (same revision, no network needed).
    local=encoder.get('local_path')
    source,revision=(local,None) if local and Path(local).exists() else (encoder['id'],encoder['revision'])
    tokenizer=AutoTokenizer.from_pretrained(source,revision=revision)
    model=AutoModel.from_pretrained(source,revision=revision).to(device).eval()
    lock=threading.Lock(); app=FastAPI(); expected=os.environ.get('RETRIEVAL_API_KEY','')
    def authenticate(request):
        if expected and not hmac.compare_digest(request.headers.get('Authorization',''),'Bearer '+expected):
            raise HTTPException(status_code=401,detail='Unauthorized')
    @app.get('/health')
    def health(): return {'ok':True,'documents':len(corpus),'encoder':encoder,'source':info}
    @app.post('/retrieve')
    async def retrieve(request: Request):
        authenticate(request); payload=await request.json()
        queries=payload.get('queries'); k=payload.get('topk',3)
        if not isinstance(queries,list) or not 1<=len(queries)<=16 or any(not isinstance(q,str) or not q.strip() or len(q)>4096 for q in queries):
            raise HTTPException(400,'Provide 1-16 nonempty string queries, at most 4096 characters each')
        if not isinstance(k,int) or not 1<=k<=10: raise HTTPException(400,'topk must be between 1 and 10')
        # Serialized access also protects the corpus file cursor. Use one server worker.
        with lock,torch.no_grad():
            inputs=tokenizer(['query: '+q for q in queries],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
            h=model(**inputs).last_hidden_state
            masked=h.masked_fill(~inputs.attention_mask[...,None].bool(),0.)
            emb=masked.sum(1)/inputs.attention_mask.sum(1)[:,None]
            emb=torch.nn.functional.normalize(emb,dim=-1).cpu().numpy().astype(np.float32,order='C')
            scores,indices=index.search(emb,k)
            results=[[{'document':corpus[int(i)],'score':float(s)} for i,s in zip(ids,ss) if i>=0]
                     for ids,ss in zip(indices,scores)]
        return {'result':results}
    return app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True); p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8000); p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args()
    if a.host not in {'127.0.0.1','localhost'} and not os.environ.get('RETRIEVAL_API_KEY'):
        p.error('Set RETRIEVAL_API_KEY before exposing the service beyond localhost')
    import uvicorn
    uvicorn.run(create_app(a.assets,a.device),host=a.host,port=a.port,workers=1)

if __name__=='__main__': main()
