"""One-command dependency-checked research pipeline for a complete dataset."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pairs',required=True)
    p.add_argument('--out',required=True);p.add_argument('--g-steps',type=int,default=10000)
    p.add_argument('--risk-steps',type=int,default=5000);p.add_argument('--ngf',type=int,default=64)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--budget',type=float,default=.10)
    a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    stages=[]
    def run(name,*args):
        command=[sys.executable,'-m','expression.cli',*map(str,args)]
        print('STAGE',name,flush=True)
        result=subprocess.run(command,check=False)
        stages.append(dict(stage=name,exit_code=result.returncode))
        (out/'pipeline_status.json').write_text(json.dumps(stages,indent=2))
        if result.returncode:raise SystemExit('Automatic pipeline stopped at '+name+'; see the explicit error above.')
    manifest=out/'manifest.jsonl'
    run('prepare','prepare','--pairs',a.pairs,'--out',manifest,'--auto-split-groups')
    for model in ['cut','expression_cut']:
        run('train_'+model,'train','--manifest',manifest,'--out',out/model,'--model',model,
            '--steps',a.g_steps,'--ngf',a.ngf,'--batch-size',a.batch_size)
    g=out/'expression_cut/generator.pt';risk=out/'risk/risk.pt'
    run('challenge','challenge','--manifest',manifest,'--out',out/'challenge')
    run('fit_risk','fit-risk','--manifest',manifest,'--generator',g,'--out',out/'risk','--steps',a.risk_steps)
    run('certify','certify','--manifest',manifest,'--generator',g,'--risk',risk,'--out',out/'calibration','--budget',a.budget)
    run('test','evaluate','--manifest',manifest,'--generator',g,'--risk',risk,
        '--certificate',out/'calibration/certificate.json','--out',out/'test')
    for model in ['cut','expression_cut']:
        run('held_out_'+model,'evaluate-generator','--manifest',manifest,
            '--generator',out/model/'generator.pt','--out',out/'test'/model)


if __name__=='__main__':main()
