"""Search-R1 HTTP retrieval contract. No hidden live-web or synthetic fallback."""
from dataclasses import dataclass
import hashlib
import html
import json
import os
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

@dataclass
class Passage:
    id: str
    contents: str
    score: float | None = None

class HTTPRetriever:
    def __init__(self, endpoint: str, *, corpus_id: str, retriever_id: str, timeout: float=30.):
        u=urlsplit(endpoint)
        if u.scheme not in {'http','https'} or not u.netloc:
            raise ValueError('A full http(s) retrieval URL is required')
        if u.username or u.password or u.query:
            raise ValueError('Do not embed credentials in retrieval URLs')
        if not corpus_id.strip() or not retriever_id.strip():
            raise ValueError('Record the actual corpus and retriever revisions')
        self.endpoint,self.timeout=endpoint,timeout
        self.provenance={'endpoint':endpoint,'corpus_id':corpus_id,'retriever_id':retriever_id}

    def search(self,query: str,top_k: int) -> list[Passage]:
        if not query.strip() or top_k<1: raise ValueError('Invalid retrieval request')
        body=json.dumps({'queries':[query],'topk':top_k,'return_scores':True}).encode()
        headers={'Content-Type':'application/json'}
        if os.environ.get('RETRIEVAL_API_KEY'): headers['Authorization']='Bearer '+os.environ['RETRIEVAL_API_KEY']
        req=Request(self.endpoint,data=body,headers=headers,method='POST')
        with urlopen(req,timeout=self.timeout) as response:
            raw=response.read(8_000_001)
        if len(raw)>8_000_000: raise ValueError('Retrieval response too large')
        return parse_response(json.loads(raw),top_k)

def parse_response(data: dict, top_k: int) -> list[Passage]:
    result=data.get('result')
    if not isinstance(result,list) or len(result)!=1 or not isinstance(result[0],list):
        raise ValueError('Expected Search-R1 API {"result":[[documents...]]}')
    passages=[]
    for item in result[0][:top_k]:
        doc=item.get('document',item)
        text=doc.get('contents')
        if not isinstance(text,str): raise ValueError('A retrieved document has no contents string')
        ident=str(doc.get('id',hashlib.sha256(text.encode()).hexdigest()[:16]))
        score=item.get('score')
        passages.append(Passage(ident,text,float(score) if score is not None else None))
    return passages

def format_passages(passages: list[Passage]) -> str:
    # Escape control-tag-like text supplied by the environment, not by the policy.
    lines=[]
    for i,p in enumerate(passages,1):
        title,_,body=p.contents.partition('\n')
        lines.append(f'Doc {i}(Title: {html.escape(title,quote=False)}) {html.escape(body,quote=False)}')
    return '\n'.join(lines) or 'No passages were returned.'
