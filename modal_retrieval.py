"""Download the Wikipedia-2018 E5 retrieval assets into a dedicated Modal volume.

CPU only (2 cores, 4 GiB). Run detached so it survives a closed laptop:

    modal run --detach modal_retrieval.py

Nothing runs merely by importing this file.
"""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent
VOLUME_NAME = 'info7375-ghouse-retrieval'
app = modal.App('info7375-ghouse-retrieval-download')
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('huggingface_hub==0.30.2', 'requests==2.32.3')
         .add_local_file(ROOT / 'scripts' / 'download_retrieval_assets.py', '/app/download_retrieval_assets.py')
         .add_local_file(ROOT / 'search_r1_team' / 'corpus.py', '/app/corpus.py'))


@app.function(image=image, cpu=2, memory=4096, timeout=6 * 3600, volumes={'/retrieval': volume})
def download():
    import subprocess
    try:
        subprocess.run(['python', '/app/download_retrieval_assets.py', '--output', '/retrieval/wiki18-e5',
                        '--acknowledge-large-download'], check=True)
    finally:
        volume.commit()


@app.function(image=image, cpu=2, memory=4096, timeout=3600, volumes={'/retrieval': volume})
def index_corpus():
    """Build the corpus line-offset file once (a 14 GB scan) so GPU containers start fast."""
    import sys, time
    sys.path.insert(0, '/app')
    from corpus import IndexedCorpus
    start = time.monotonic()
    corpus = IndexedCorpus('/retrieval/wiki18-e5/wiki-18.jsonl')
    first, last = corpus[0], corpus[len(corpus) - 1]
    corpus.close()
    volume.commit()
    return {'rows': len(corpus), 'seconds': round(time.monotonic() - start), 'first_id': first['id'], 'last_id': last['id']}


@app.local_entrypoint()
def main(job: str = 'download'):
    print(download.remote() if job == 'download' else index_corpus.remote())
