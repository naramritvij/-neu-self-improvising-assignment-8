# Retrieval setup

The search client expects the Search-R1 HTTP contract:

```json
{"queries": ["a model-generated query"], "topk": 3, "return_scores": true}
```

The response is `{"result": [[{"document": {"id": "...", "contents": "Title\nText"}, "score": 0.1}]]}`.
This is a contract example, not a measured search result.

## Prefer an existing class endpoint

Ask whether the class already provides a Wikipedia-2018/E5 retrieval service.
Verify its health, authentication, model/corpus revisions and compatibility from
the GPU container before loading Qwen. Record that provenance for both algorithms.
Do not replace it silently with a live web API or synthetic passages.

## Self-hosted alternative

The repository includes a CPU-FAISS service. The full assets require substantial
storage/RAM; inspect source sizes and workspace capacity before downloading.

```bash
python -m pip install -r requirements-retriever.txt
python scripts/download_retrieval_assets.py --output data/retrieval --acknowledge-large-download
python scripts/serve_retrieval.py --assets data/retrieval --host 127.0.0.1 --port 8000
```

Install a suitable PyTorch build separately. The downloader records source/encoder
revisions and hashes. The server checks index/corpus row alignment, uses E5 query
prefixes and normalized mean pooling, and serializes access to its corpus cursor.
The full index and actual E5 integration have not been run here.

The default server binds locally. To expose it beyond localhost, set a strong
`RETRIEVAL_API_KEY`, use the appropriate host/network configuration and deploy
behind HTTPS/access controls. This script is not an automatically hosted Modal
endpoint. Do not put an unprotected service on the public internet.

Modal containers cannot reach your laptop's service through their own
`127.0.0.1`. Use a properly hosted, reachable URL. Supply the same URL/corpus/
encoder provenance to both training and evaluation and keep credentials in secrets,
not URLs. Retrieval failures stop the run rather than fabricate information.

Sources:
- https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py
- https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index
- https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus
