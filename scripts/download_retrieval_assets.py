"""Download the paper's Wikipedia-2018 E5 retrieval assets (~70 GB) by streaming.

The two flat-index parts are streamed straight into one `e5_Flat.index` (upstream concatenates
part_aa and part_ab, scripts/download.py), and the corpus is decompressed on the fly, so no part
or archive is kept on disk. SHA-256 digests are computed while writing. Self-contained: needs only
`huggingface_hub` and `requests`.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

INDEX_REPO = 'PeterJinGo/wiki-18-e5-index'
CORPUS_REPO = 'PeterJinGo/wiki-18-corpus'
ENCODER = 'intfloat/e5-base-v2'
CHUNK = 8 * 1024 * 1024


def stream(url: str, headers: dict, session, retries: int = 5):
    for attempt in range(retries):
        try:
            response = session.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
            response.raise_for_status()
            return response
        except Exception as exc:  # network errors are retried with backoff
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f'  download error ({type(exc).__name__}); retrying in {wait}s', flush=True)
            time.sleep(wait)


def write_parts(urls, target: Path, headers, session) -> tuple[str, int]:
    digest, total, start = hashlib.sha256(), 0, time.monotonic()
    with target.open('wb') as out:
        for url in urls:
            with stream(url, headers, session) as response:
                for chunk in response.iter_content(CHUNK):
                    out.write(chunk); digest.update(chunk); total += len(chunk)
                    if total % (5 * 1024 ** 3) < CHUNK:
                        rate = total / max(1e-6, time.monotonic() - start) / 1e6
                        print(f'  {target.name}: {total / 1e9:.1f} GB at {rate:.0f} MB/s', flush=True)
    return digest.hexdigest(), total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--acknowledge-large-download', action='store_true')
    a = p.parse_args()
    if not a.acknowledge_large_download:
        p.error('About 70 GB will be written; add --acknowledge-large-download')
    import requests
    from huggingface_hub import HfApi, hf_hub_url, snapshot_download
    from huggingface_hub.utils import build_hf_headers
    api, session, headers = HfApi(), requests.Session(), build_hf_headers()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest_path = a.output / 'assets_manifest.json'
    if manifest_path.exists():
        p.error('Assets are already frozen here; use the saved manifest or a new directory')
    manifest = {
        'index': {'id': INDEX_REPO, 'revision': api.dataset_info(INDEX_REPO).sha},
        'corpus': {'id': CORPUS_REPO, 'revision': api.dataset_info(CORPUS_REPO).sha},
        'encoder': {'id': ENCODER, 'revision': api.model_info(ENCODER).sha},
    }
    started = time.monotonic()

    index = a.output / 'e5_Flat.index'
    urls = [hf_hub_url(INDEX_REPO, name, repo_type='dataset', revision=manifest['index']['revision'])
            for name in ('part_aa', 'part_ab')]
    print('Index: streaming part_aa + part_ab', flush=True)
    manifest['index_sha256'], manifest['index_bytes'] = write_parts(urls, index, headers, session)

    corpus = a.output / 'wiki-18.jsonl'
    url = hf_hub_url(CORPUS_REPO, 'wiki-18.jsonl.gz', repo_type='dataset', revision=manifest['corpus']['revision'])
    print('Corpus: streaming and decompressing wiki-18.jsonl.gz', flush=True)
    digest, total, rows = hashlib.sha256(), 0, 0
    with stream(url, headers, session) as response, corpus.open('wb') as out:
        response.raw.decode_content = True
        with gzip.GzipFile(fileobj=response.raw) as source:
            while chunk := source.read(CHUNK):
                out.write(chunk); digest.update(chunk); total += len(chunk); rows += chunk.count(b'\n')
    manifest['corpus_sha256'], manifest['corpus_bytes'], manifest['corpus_rows'] = digest.hexdigest(), total, rows

    print('Encoder: snapshot of intfloat/e5-base-v2 (PyTorch weights only)', flush=True)
    encoder_dir = a.output / 'e5-base-v2'
    snapshot_download(ENCODER, revision=manifest['encoder']['revision'], local_dir=encoder_dir,
                      allow_patterns=['*.json', '*.txt', 'model.safetensors'])
    manifest['encoder']['local_path'] = str(encoder_dir)
    manifest['download_seconds'] = round(time.monotonic() - started)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
