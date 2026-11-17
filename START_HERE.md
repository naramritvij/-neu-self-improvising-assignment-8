# Start here — Assignment 8

**Repository:** `naramritvij/-neu-self-improvising-assignment-8` · **Authors:** Ritvij Naram and Ghouse

The results, method and cost are in [README.md](README.md). Paper-parity decisions, with upstream
file and line citations, are in [docs/PAPER_PARITY.md](docs/PAPER_PARITY.md).

## Local checks

```bash
python scripts/preflight.py
python -m unittest discover -s tests -v
```

The tests use synthetic CPU fixtures and download no model or dataset.

## Cloud runs

See [docs/MODAL_SETUP.md](docs/MODAL_SETUP.md) and the "Reproduce" section of the README. Every
GPU job requires `--acknowledge-compute`; nothing runs on push.
