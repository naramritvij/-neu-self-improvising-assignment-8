"""Download the paper's Wikipedia-2018 E5 assets. Large storage/memory requirements."""
import argparse
import gzip
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from search_r1_team.runtime import sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acknowledge-large-download',action='store_true')
    a=p.parse_args()
    if not a.acknowledge_large_download: p.error('Inspect upstream asset sizes and available disk/RAM, then add --acknowledge-large-download')
    from huggingface_hub import HfApi,hf_hub_download
    api=HfApi(); a.output.mkdir(parents=True,exist_ok=True)
    target=a.output/'assets_manifest.json'
    if target.exists(): p.error('Assets are already frozen here; use the saved manifest or a new directory')
    repos={'index':'PeterJinGo/wiki-18-e5-index','corpus':'PeterJinGo/wiki-18-corpus'}
    manifest={k:{'id':v,'revision':api.dataset_info(v).sha} for k,v in repos.items()}
    paths=[]
    for name in ('part_aa','part_ab'):
        info=manifest['index']
        paths.append(Path(hf_hub_download(repo_id=info['id'],repo_type='dataset',revision=info['revision'],filename=name,local_dir=a.output/'downloads')))
    info=manifest['corpus']
    archive=hf_hub_download(repo_id=info['id'],repo_type='dataset',revision=info['revision'],filename='wiki-18.jsonl.gz',local_dir=a.output/'downloads')
    index=a.output/'e5_Flat.index'; corpus=a.output/'wiki-18.jsonl'
    for dest in (index,corpus):
        if dest.exists(): p.error('An unfinished output exists. Inspect it; do not silently overwrite large assets')
    with index.open('wb') as output:
        for piece in paths:
            with piece.open('rb') as f: shutil.copyfileobj(f,output,length=1024*1024)
    with gzip.open(archive,'rb') as source,corpus.open('wb') as output:
        shutil.copyfileobj(source,output,length=1024*1024)
    manifest['encoder']={'id':'intfloat/e5-base-v2','revision':api.model_info('intfloat/e5-base-v2').sha}
    manifest['index_sha256']=sha256(index); manifest['corpus_sha256']=sha256(corpus)
    manifest['index_bytes']=index.stat().st_size; manifest['corpus_bytes']=corpus.stat().st_size
    write_json(target,manifest); print(target)

if __name__=='__main__': main()
