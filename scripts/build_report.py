"""Build an honest comparison report; create plots only from completed real run files."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from search_r1_team.runtime import sha256

LABELS={'direct':'Direct inference','own_grpo':'Own LoRA + GRPO','own_ppo':'Own LoRA + PPO',
        'released_grpo':'Released GRPO checkpoint','released_ppo':'Released PPO checkpoint'}
PAPER={'direct':2.4,'own_grpo':23.2,'own_ppo':26.4,'released_grpo':23.2,'released_ppo':26.4}

def collect(runs_root):
    results=[]; shared=None; retrieval_shared=None
    for method in LABELS:
        folder=Path(runs_root)/method; metrics=folder/'metrics.json'; manifest=folder/'manifest.json'
        if not metrics.exists() and not manifest.exists(): continue
        if not metrics.exists() or not manifest.exists(): raise ValueError(f'{method}: missing metrics or run manifest')
        m=json.loads(metrics.read_text()); run=json.loads(manifest.read_text())
        if run.get('status')!='completed' or run.get('coverage')!='full' or m.get('coverage')!='complete':
            raise ValueError(f'{method}: incomplete/diagnostic run cannot be used in the final table')
        if run['signature']['method']!=method: raise ValueError(f'{method}: method label mismatch')
        if m['method']!=method.split('_')[-1]: raise ValueError(f'{method}: evaluator method mismatch')
        if sha256(folder/'predictions.jsonl')!=m['predictions_sha256']: raise ValueError('Predictions changed after scoring')
        if run['total_questions']!=m['expected_questions'] or m['evaluated_questions']!=run['total_questions']:
            raise ValueError('Incomplete benchmark denominator')
        signature=(run['signature']['questions_sha256'],m['gold_sha256'],m['expected_questions'])
        if shared is not None and signature!=shared: raise ValueError('Cannot compare different held-out datasets')
        shared=signature
        if method!='direct':
            # Do not call different retrieval settings a controlled algorithm comparison.
            retrieval=run['signature']['retrieval']
            if retrieval_shared is not None and retrieval!=retrieval_shared:
                raise ValueError('Search methods used different retrieval provenance')
            retrieval_shared=retrieval
        if method.startswith('own_'):
            training=run['signature'].get('own_training',{})
            if training.get('run_kind')!='experiment' or training.get('step',0)<=2:
                raise ValueError('Smoke-only adapters cannot populate the final experiment table')
        results.append({'method':method,'label':LABELS[method],'metrics':m,'manifest':run})
    return results


def plot_results(results,figdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,5))
    bars=ax.bar([x['label'] for x in results],[x['metrics']['em_percent'] for x in results])
    ax.bar_label(bars,fmt='%.1f%%'); ax.set_ylabel('Exact match (%)'); ax.set_ylim(0,100)
    ax.set_title('Measured Bamboogle results — actual completed runs only')
    ax.tick_params(axis='x',rotation=15); fig.tight_layout(); fig.savefig(figdir/'bamboogle_em.png',dpi=160); plt.close(fig)


def plot_training(path,algorithm,figdir):
    if not path: return []
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('status')!='completed' or manifest.get('run_kind')!='experiment':
        raise ValueError('Only completed, non-smoke training logs belong in the experiment report')
    if manifest['config']['algorithm']!=algorithm: raise ValueError('Training log algorithm mismatch')
    rows=[json.loads(s) for s in (path/'training.jsonl').read_text().splitlines() if s.strip()]
    if not rows or rows[-1]['step']!=manifest['completed_steps']: raise ValueError('Incomplete training logs')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[]
    for field,title in [('reward_mean','Training exact-match reward'),('validation_em','Held-out NQ/HotpotQA validation EM')]:
        points=[x for x in rows if field in x]
        if not points: continue
        fig,ax=plt.subplots(figsize=(8,4)); ax.plot([r['step'] for r in points],[r[field] for r in points])
        ax.set_xlabel('Optimizer step'); ax.set_ylabel(title); ax.set_title(algorithm.upper()+' — '+title)
        ax.set_ylim(0,1); fig.tight_layout()
        name=f'{algorithm}_{field}.png'; figdir.mkdir(parents=True,exist_ok=True)
        fig.savefig(figdir/name,dpi=160); plt.close(fig); files.append(name)
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,default=ROOT/'runs/evaluation')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results')
    p.add_argument('--grpo-training',type=Path); p.add_argument('--ppo-training',type=Path)
    p.add_argument('--no-plots',action='store_true')
    a=p.parse_args(); results=collect(a.runs_root); a.output_dir.mkdir(parents=True,exist_ok=True)
    measured={x['method']:x for x in results}; lines=[
      '# Search-R1 replication — experiment report','',
      '**Team:** Ritvij and Ghouse. **Course:** INFO 7375, Week 08.','',
      '**Status:** '+('Measured runs are listed below; inspect missing rows and parity limitations before submission.' if results else
                         'NOT SUBMISSION-READY: no Qwen/Bamboogle experiment results have been produced.'),'',
      '## Results','',
      'Published references are from arXiv 2503.09516v5, Tables 2–3. They are NOT this project’s measurements and are NOT published LoRA scores.','',
      '| Experiment | Paper reference EM | Measured EM | Correct / evaluated | Status |',
      '|---|---:|---:|---:|---|']
    for method,label in LABELS.items():
        if method in measured:
            m=measured[method]['metrics']; lines.append(f'| {label} | {PAPER[method]:.1f}% | {m["em_percent"]:.1f}% | {m["correct"]} / {m["evaluated_questions"]} | Completed evaluation |')
        else: lines.append(f'| {label} | {PAPER[method]:.1f}% | — | — | Not run |')
    lines += ['', '## Methodology', '',
      'The prepared implementation uses frozen Qwen2.5-3B-Instruct weights and rank-16 LoRA adapters. '
      'GRPO uses grouped answer rewards. PPO uses a separate LoRA critic and a value head, GAE and clipped updates. '
      'Retrieved text remains in the causal context but is excluded from action losses. '
      'Bamboogle labels are available only to the scorer, not to inference or training. '
      'The NQ/HotpotQA training subset and separate validation subset are frozen before training.', '',
      '## Reproduction limits', '',
      'See ../docs/PAPER_PARITY.md. This is a compact, single-GPU educational adaptation, not a byte-for-byte veRL replication. '
      'Paper-scale training, prompt equality, dependency compatibility and complete GPU behavior have not been established by CPU unit tests. '
      'Released-checkpoint evaluation is distinct from training a new LoRA adapter.', '',
      '## Interpretation', '',
      ('Do not conclude that one algorithm is better from one seed or unmatched compute. Inspect per-question errors, search failures, '
       'actual rollout/token budgets, validation curves and any deviations before writing the final discussion.' if results else
       'No conclusions about model quality or GRPO versus PPO can be drawn before the real experiments run.'), '',
      '## Completion evidence still required', '',
      'Class-starter reproduction log; successful Qwen GPU smoke runs; actual trained adapters; full held-out predictions; '
      'retriever/model/data revision records; real result figures; error analysis; and a truthful contribution log.', '']
    if results and not a.no_plots:
        plot_results(results,a.output_dir/'figures')
        lines+=['## Figures','','![Measured Bamboogle EM](figures/bamboogle_em.png)','']
    if not a.no_plots:
        for alg,path in [('grpo',a.grpo_training),('ppo',a.ppo_training)]:
            for name in plot_training(path,alg,a.output_dir/'figures'):
                lines += [f'![{alg.upper()} actual training log](figures/{name})','']
    (a.output_dir/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(a.output_dir/'REPORT.md')

if __name__=='__main__': main()
