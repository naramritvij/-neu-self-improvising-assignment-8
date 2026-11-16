"""Score actual saved predictions; no model execution and no invented results."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search_r1_team.data import read_jsonl
from search_r1_team.metrics import exact_match, extract_answer


def score_records(gold: list[dict], predictions: list[dict], *, method: str, allow_partial: bool=False) -> dict:
    if method not in {'direct','grpo','ppo'}:
        raise ValueError('Unknown method.')
    if not gold or not predictions:
        raise ValueError('Gold and predictions must both be nonempty.')
    expected = {r['id']:r for r in gold}
    predicted = {r['id']:r for r in predictions}
    if len(expected) != len(gold) or len(predicted) != len(predictions):
        raise ValueError('Duplicate IDs are not allowed.')
    extra, missing = set(predicted)-set(expected), set(expected)-set(predicted)
    if extra:
        raise ValueError(f'{len(extra)} prediction IDs are not in the frozen dataset.')
    if missing and not allow_partial:
        raise ValueError(f'{len(missing)} predictions missing; --allow-partial is for labelled diagnostics only.')
    for row in gold:
        if not isinstance(row.get('answers'), list):
            raise ValueError('Gold records must contain an answers list.')
        exact_match('', row['answers'])
    correct = invalid = 0
    for row in predictions:
        if not isinstance(row.get('prediction'), str):
            raise ValueError(f'{row["id"]}: prediction must be a string; use empty string for a failed attempt.')
        answer = extract_answer(row['prediction'], tagged=method != 'direct')
        invalid += int(not answer)
        correct += exact_match(answer, expected[row['id']]['answers'])
    n = len(predictions)
    return {'method':method,'metric':'normalized_exact_match',
        'evaluator_parity_with_upstream':'normalization_inspected_response_only_extraction; end_to_end_paper_parity_not_claimed',
        'coverage':'partial_diagnostic' if missing else 'complete',
        'expected_questions':len(gold),'evaluated_questions':n,
        'missing_predictions':len(missing),'invalid_or_empty_answers':invalid,
        'correct':correct,'em':correct/n,'em_percent':100*correct/n,'score_denominator':n}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--method', choices=['direct','grpo','ppo'], required=True)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.output.exists() and not args.overwrite:
        p.error('Output exists; choose a new path or explicitly pass --overwrite.')
    if args.output.resolve() in {args.gold.resolve(), args.predictions.resolve()}:
        p.error('Output must not overwrite an input.')
    try:
        result = score_records(read_jsonl(args.gold), read_jsonl(args.predictions),
                               method=args.method, allow_partial=args.allow_partial)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        p.error(str(exc))
    result.update({'evaluated_at_utc':datetime.now(timezone.utc).isoformat(),
        'gold_sha256':hashlib.sha256(args.gold.read_bytes()).hexdigest(),
        'predictions_sha256':hashlib.sha256(args.predictions.read_bytes()).hexdigest()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
