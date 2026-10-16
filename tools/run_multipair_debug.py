"""Reproduce the six-pair bug investigation, retaining same-source demo labels."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from expression.manifest import load_pair_csv,prepare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',default=str(ROOT/'examples/multipair/pairs.csv'))
    p.add_argument('--out',default=str(ROOT/'artifacts/multipair_debug_rerun'))
    p.add_argument('--steps',type=int,default=160);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    pairs=load_pair_csv(a.pairs);roles={r['pair_id']:r['debug_partition'] for r in pairs}
    if any(r['split']!='demo' or r['group_kind']!='unknown' for r in pairs):
        raise ValueError('This investigation deliberately uses unknown-source demo data, not patient-level test splits.')
    rows=prepare(pairs,out/'v2_full_manifest.jsonl')
    for row in rows:
        row['debug_partition']=roles[row['pair_id']]
        row['partition_scope']='image-pair debugging only; source IDs unverified'
    for role in ['full','train','dev','holdout']:
        selected=rows if role=='full' else [r for r in rows if r['debug_partition']==role]
        (out/(role+'_manifest.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in selected))
    (out/'pairs.json').write_text(json.dumps(pairs,indent=2))
    legacy=ROOT/'artifacts/multipair_debug/v1_source'
    if not legacy.is_dir():raise ValueError('Frozen v1 implementation is missing.')
    env=os.environ.copy();legacy_env=env.copy()
    legacy_env['PYTHONPATH']=str(legacy)+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-m','expression.cli','prepare','--pairs',str(Path(a.pairs).resolve()),
                    '--out',str(out/'v1_full_manifest.jsonl')],cwd=legacy,env=legacy_env,check=True)
    for variant in ['v1','v2','compat']:
        child_env=legacy_env if variant=='v1' else env.copy()
        if variant!='v1':child_env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
        target=out/('v2_compat' if variant=='compat' else variant)
        command=[sys.executable,str(ROOT/'tools/multipair_debug_worker.py'),'--variant',variant,
                 '--manifests',str(out),'--out',str(target),'--steps',str(a.steps)]
        with (out/(variant+'_run.log')).open('w') as log:
            subprocess.run(command,cwd=legacy if variant=='v1' else ROOT,env=child_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Two generator training runs completed; compatibility branch reuses v1 G weights and retrains h only.')
    print('Next: python -m tools.report_multipair_debug --base',out)


if __name__=='__main__':main()
