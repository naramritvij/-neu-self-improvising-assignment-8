"""Validate the course CSV and exclude non-QA metadata."""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
from typing import Any


def load_bamboogle_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.reader(io.StringIO(text.lstrip('\ufeff')))
    try:
        header = [cell.strip() for cell in next(reader)]
    except StopIteration as exc:
        raise ValueError('CSV is empty.') from exc
    if header.count('Question') != 1 or header.count('Answer') != 1:
        raise ValueError('Expected Question and Answer headers; the response may be a login page.')
    qi = header.index('Question')
    ais = [header.index('Answer')]
    if 'Answer 2' in header:
        ais.append(header.index('Answer 2'))
    records = []
    for sheet_row, row in enumerate(reader, start=2):
        def get(index: int) -> str:
            return row[index].strip() if index < len(row) else ''
        question = get(qi)
        answers = list(dict.fromkeys(get(i) for i in ais if get(i)))
        if not question and not answers:
            continue
        if not question or not answers:
            raise ValueError(f'Sheet row {sheet_row}: missing question or accepted answer.')
        records.append({'id':f'bamboogle_{sheet_row-1:04d}', 'question':question, 'answers':answers})
    if not records:
        raise ValueError('No question/answer records were found.')
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records, seen = [], set()
    with path.open(encoding='utf-8-sig') as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{number}: invalid JSON') from exc
            if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'].strip():
                raise ValueError(f'{path}:{number}: a nonempty string id is required')
            if row['id'] in seen:
                raise ValueError(f'{path}:{number}: duplicate id {row["id"]}')
            seen.add(row['id'])
            records.append(row)
    if not records:
        raise ValueError(f'{path}: no records')
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
