"""Run the exact fetched class starter, with documented output/budget-only overrides.

Requires the class starter's own dependency environment, not requirements.txt
from the compact Search-R1 implementation. No source is overwritten.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
PIN='3659acfad61dab2f90fa46294566ff6808452e87'

class Overrides(ast.NodeTransformer):
    def __init__(self,output,steps,smoke): self.output,self.steps,self.smoke=output,steps,smoke
    def visit_Assign(self,node):
        self.generic_visit(node)
        names=[x.id for x in node.targets if isinstance(x,ast.Name)]
        if 'output_dir' in names: node.value=ast.Constant(self.output)
        if self.smoke and 'max_completion_length' in names: node.value=ast.Constant(128)
        if self.smoke and 'eval_steps' in names: node.value=ast.Constant(2)
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='GRPOConfig':
            changes={'max_steps':self.steps}
            if self.smoke:
                changes.update(eval_strategy='no',eval_on_start=False,num_generations=2,
                    per_device_train_batch_size=2,gradient_accumulation_steps=1,use_vllm=False)
            for kw in node.keywords:
                if kw.arg in changes: kw.value=ast.Constant(changes[kw.arg])
        return node

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    source=ROOT/'vendor/tiny-zero-lora'
    if not (source/'train.py').exists(): p.error('Run scripts/fetch_starter.py first')
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=PIN: p.error('The starter source is not at the pinned class commit')
    if a.output.exists(): p.error('Choose a new output directory to preserve previous evidence')
    if a.steps<1: p.error('Steps must be positive')
    a.output.mkdir(parents=True)
    # Catch incompatible environments before a paid run, while not silently installing packages.
    from importlib.metadata import version
    import re
    expected={}
    for line in (source/'requirements.txt').read_text().splitlines():
        match=re.match(r'([\w_-]+)(?:\[[^]]+\])?==([^\s#]+)',line.strip())
        if match: expected[match[1]]=match[2]
    actual={name:version(name) for name in expected}
    if actual!=expected: p.error('Starter dependency pins differ. Use its separate environment; do not mix trainer stacks.')
    steps=2 if a.smoke else a.steps
    (a.output/'starter_manifest.json').write_text(json.dumps({
        'upstream_commit':commit,'steps':steps,'smoke_only':a.smoke,'dependencies':actual,
        'changes':'output path plus declared step budget; smoke also reduces generation/batch and disables vLLM/eval'},indent=2))
    tree=Overrides(str(a.output/'model'),steps,a.smoke).visit(ast.parse((source/'train.py').read_text()))
    ast.fix_missing_locations(tree)
    sys.path.insert(0,str(source))
    scope={'__name__':'class_starter','__file__':str(source/'train.py')}
    exec(compile(tree,str(source/'train.py'),'exec'),scope)
    scope['main']()

if __name__=='__main__': main()
