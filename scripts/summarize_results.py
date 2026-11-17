"""Summarize downloaded Modal runs into results/summary.json and results/training_curves.png.

Usage: python scripts/summarize_results.py --runs downloaded-runs
Each evaluation folder holds manifest.json, metrics.json, predictions.jsonl and traces/; each
training folder holds manifest.json and training.jsonl. Missing runs are reported as missing.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = {'direct': 0.024, 'released_grpo': 0.232, 'released_ppo': 0.264, 'own_grpo': 0.232, 'own_ppo': 0.264}
EVALS = {'direct': 'direct-v1', 'released_grpo': 'released-grpo-v02b', 'released_ppo': 'released-ppo-v02b',
         'own_grpo': 'eval-own-grpo', 'own_ppo': 'eval-own-ppo'}
TRAIN = {'grpo': 'grpo-short', 'ppo': 'ppo-short'}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    a = p.parse_args()
    out = {'evaluations': {}, 'training': {}}
    for method, folder in EVALS.items():
        d = a.runs / folder
        if not (d / 'metrics.json').exists():
            out['evaluations'][method] = {'status': 'missing'}; continue
        m = json.loads((d / 'metrics.json').read_text())
        preds = read_jsonl(d / 'predictions.jsonl')
        stops = {}
        for r in preds: stops[r['stop_reason']] = stops.get(r['stop_reason'], 0) + 1
        searches = []
        for t in (d / 'traces').glob('*.json'):
            events = json.loads(t.read_text(encoding='utf-8'))['events']
            searches.append(sum(e['kind'] == 'retrieval' for e in events))
        out['evaluations'][method] = {
            'run': folder, 'em': m['em'], 'correct': m['correct'], 'n': m['evaluated_questions'],
            'paper_em': PAPER[method], 'invalid_or_empty_answers': m['invalid_or_empty_answers'],
            'stop_reasons': stops, 'mean_searches': round(sum(searches) / len(searches), 2) if searches else 0.0,
            'seconds_per_question': round(sum(r['seconds'] for r in preds) / len(preds), 2)}
    curves = {}
    for algo, folder in TRAIN.items():
        d = a.runs / folder
        if not (d / 'training.jsonl').exists():
            out['training'][algo] = {'status': 'missing'}; continue
        log = read_jsonl(d / 'training.jsonl'); manifest = json.loads((d / 'manifest.json').read_text())
        curves[algo] = log
        out['training'][algo] = {
            'run': folder, 'status': manifest['status'], 'steps': len(log),
            'rollouts_per_step': log[0]['rollouts'], 'total_rollouts': sum(r['rollouts'] for r in log),
            'mean_seconds_per_step': round(sum(r['seconds'] for r in log) / len(log), 1),
            'reward_first5': round(sum(r['reward_mean'] for r in log[:5]) / min(5, len(log)), 3),
            'reward_last5': round(sum(r['reward_mean'] for r in log[-5:]) / min(5, len(log)), 3),
            'mean_searches_first5': round(sum(r['mean_searches'] for r in log[:5]) / min(5, len(log)), 2),
            'mean_searches_last5': round(sum(r['mean_searches'] for r in log[-5:]) / min(5, len(log)), 2),
            'actor_trainable_parameters': manifest.get('actor_trainable_parameters'),
            'critic_trainable_parameters': manifest.get('critic_trainable_parameters')}
    results = ROOT / 'results'; results.mkdir(exist_ok=True)
    (results / 'summary.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=110)
        for algo, log in curves.items():
            steps = [r['step'] for r in log]
            k = 5
            smooth = lambda xs: [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]
            axes[0].plot(steps, smooth([r['reward_mean'] for r in log]), label=f'{algo.upper()} (5-step mean)')
            axes[1].plot(steps, smooth([r['mean_searches'] for r in log]), label=algo.upper())
        axes[0].set_title('Training reward (exact match on NQ/HotpotQA rollouts)'); axes[0].set_xlabel('step')
        axes[1].set_title('Searches per rollout'); axes[1].set_xlabel('step')
        for ax in axes: ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(results / 'training_curves.png')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
