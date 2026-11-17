"""Search-R1 HTTP retrieval contract. No hidden live-web or synthetic fallback."""
from dataclasses import dataclass
import hashlib
import html
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

@dataclass
class Passage:
    id: str
    contents: str
    score: float | None = None

class HTTPRetriever:
    def __init__(self, endpoint: str, *, corpus_id: str, retriever_id: str, timeout: float=30.,
                 retries: int=5, backoff: float=1.):
        u=urlsplit(endpoint)
        if u.scheme not in {'http','https'} or not u.netloc:
            raise ValueError('A full http(s) retrieval URL is required')
        if u.username or u.password or u.query:
            raise ValueError('Do not embed credentials in retrieval URLs')
        if not corpus_id.strip() or not retriever_id.strip():
            raise ValueError('Record the actual corpus and retriever revisions')
        self.endpoint,self.timeout,self.retries,self.backoff=endpoint,timeout,retries,backoff
        self.provenance={'endpoint':endpoint,'corpus_id':corpus_id,'retriever_id':retriever_id}

    def search(self,query: str,top_k: int) -> list[Passage]:
        return self.search_batch([query],top_k)[0]

    def search_batch(self,queries: list[str],top_k: int) -> list[list[Passage]]:
        """One POST per 16 queries (the server's per-request limit), in order."""
        if not queries or top_k<1 or any(not q.strip() for q in queries): raise ValueError('Invalid retrieval request')
        out=[]
        for i in range(0,len(queries),16):
            chunk=queries[i:i+16]
            out.extend(parse_batch(self._post({'queries':chunk,'topk':top_k,'return_scores':True}),len(chunk),top_k))
        return out

    def _post(self,payload: dict) -> dict:
        body=json.dumps(payload).encode()
        headers={'Content-Type':'application/json'}
        if os.environ.get('RETRIEVAL_API_KEY'): headers['Authorization']='Bearer '+os.environ['RETRIEVAL_API_KEY']
        # Transient failures (connection reset, timeout, HTTP 5xx/429) are retried with exponential
        # backoff so one network blip cannot kill a multi-hour run. Client errors (4xx) and malformed
        # responses fail immediately: a fabricated or empty passage is never substituted.
        for attempt in range(self.retries+1):
            try:
                req=Request(self.endpoint,data=body,headers=headers,method='POST')
                with urlopen(req,timeout=self.timeout) as response:
                    raw=response.read(8_000_001)
                break
            except HTTPError as exc:
                if exc.code not in (429,500,502,503,504) or attempt==self.retries: raise
            except (URLError,TimeoutError,ConnectionError):
                if attempt==self.retries: raise
            time.sleep(self.backoff*2**attempt)
        if len(raw)>8_000_000: raise ValueError('Retrieval response too large')
        return json.loads(raw)

def parse_response(data: dict, top_k: int) -> list[Passage]:
    return parse_batch(data,1,top_k)[0]


def parse_batch(data: dict, n_queries: int, top_k: int) -> list[list[Passage]]:
    result=data.get('result')
    if not isinstance(result,list) or len(result)!=n_queries or not all(isinstance(r,list) for r in result):
        raise ValueError('Expected Search-R1 API {"result":[[documents...], ...]} with one list per query')
    return [_passages(r,top_k) for r in result]


def _passages(items: list, top_k: int) -> list[Passage]:
    passages=[]
    for item in items[:top_k]:
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
