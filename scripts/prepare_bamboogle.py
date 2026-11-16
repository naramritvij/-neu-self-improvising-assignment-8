"""Prepare held-out questions and separate labels. Never launches training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search_r1_team.data import load_bamboogle_csv, write_jsonl

SHEET_ID = '1jwcsA5kE4TObr9YHn9Gc-wQHYjTbLhDGx6tmIzMhl_U'
CSV_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='Local CSV exported from the course sheet when public download is unavailable.')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data'/'processed')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    names = ['bamboogle_questions.jsonl','bamboogle_labels.jsonl','manifest.json']
    if not args.overwrite and any((args.output_dir/f).exists() for f in names):
        parser.error('Prepared files exist; use a new output directory or explicitly pass --overwrite.')
    try:
        if args.csv:
            raw, source = args.csv.read_bytes(), str(args.csv.resolve())
        else:
            request = Request(CSV_URL, headers={'User-Agent':'INFO7375-team-preparer/1.0'})
            with urlopen(request, timeout=45) as response:
                raw = response.read(5_000_001)
            if len(raw) > 5_000_000:
                raise ValueError('CSV exceeded the safety size limit.')
            source = CSV_URL
        rows = load_bamboogle_csv(raw.decode('utf-8-sig'))
    except (OSError, URLError, UnicodeError, ValueError) as exc:
        parser.error(f'{exc}\nWhen needed, export the sheet as CSV in your browser and use --csv PATH.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions, labels = [args.output_dir/f for f in names[:2]]
    write_jsonl(questions, [{'id':r['id'],'question':r['question']} for r in rows])
    write_jsonl(labels, [{'id':r['id'],'answers':r['answers']} for r in rows])
    manifest = {
        'dataset':'Course Bamboogle sheet','split':'held_out_evaluation_only',
        'source':source,'sheet_id':SHEET_ID,'gid':0,
        'prepared_at_utc':datetime.now(timezone.utc).isoformat(),
        'num_questions':len(rows),'source_csv_sha256':hashlib.sha256(raw).hexdigest(),
        'questions_sha256':hashlib.sha256(questions.read_bytes()).hexdigest(),
        'labels_sha256':hashlib.sha256(labels.read_bytes()).hexdigest(),
        'warning':'Never pass labels to prompts, training or checkpoint selection.'}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Prepared {len(rows)} held-out questions in {args.output_dir}')
    print('Model input: bamboogle_questions.jsonl. Evaluator-only gold: bamboogle_labels.jsonl.')

if __name__ == '__main__': main()
