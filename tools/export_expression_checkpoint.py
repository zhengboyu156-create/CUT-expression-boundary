"""Wrap native train.py weights with explicit operator-provided training provenance."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from expression.manifest import read_manifest,sha256
from expression.runner import make_options
from models.expression_cut_model import ExpressionCUTModel


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--ngf',type=int,default=64);p.add_argument('--steps',type=int,required=True)
    p.add_argument('--no-antialias',action='store_true');p.add_argument('--no-antialias-up',action='store_true')
    a=p.parse_args();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    rows=[r for r in read_manifest(a.manifest) if r['split']=='train']
    if not rows:raise ValueError('No declared generator-training groups.')
    opt=make_options(a.manifest,out.parent,ngf=a.ngf,no_antialias=a.no_antialias,no_antialias_up=a.no_antialias_up)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=ExpressionCUTModel(opt);model.netG.load_state_dict(state,strict=True)
    metadata=dict(options=vars(opt),training_groups=sorted({r['group_id'] for r in rows}),demo_only=False,
                  steps=a.steps,manifest_hash=sha256(a.manifest),source_checkpoint_hash=sha256(a.checkpoint),
                  provenance_status='training-source membership supplied by operator; not inferred from weights')
    torch.save(dict(generator=model.netG.state_dict(),metadata=metadata),out)
    print('Exported strict architecture match; historical training provenance remains an operator assertion.')


if __name__=='__main__':main()
