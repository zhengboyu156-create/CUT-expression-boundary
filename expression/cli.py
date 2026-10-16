"""Run with python -m expression.cli --help. No GUI or manual ROIs required."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .manifest import prepare,load_pair_csv,automatic_group_split,read_manifest,sha256
from .challenge import fit_morphology,retrieve
from .calibration import fit_and_certify,gate,representative_per_group
from .runner import (seed_everything,make_options,make_dataset,train_generator,load_generator,
                     train_risk,load_risk,evaluate_generator,metric_summary,generated_batch)
from .statistics import foreground,to_tensor,to_rgb,components
from .protocol import policy_hash
from .artifact_io import save_png


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def predict_image(he_path,generator_path,risk_path,out,certificate_path=None,source_id='deployment'):
    """H&E-only deployment. Reference IHC is neither accepted nor accessed."""
    model,opt,gm=load_generator(generator_path)
    risk,rm=load_risk(risk_path,generator_path) if risk_path else (None,None)
    rgb=np.array(Image.open(he_path).convert('RGB'));height,width=rgb.shape[:2];w=opt.stats_window
    if width%w or height%w:raise ValueError('Deployment shape must be divisible by the frozen window.')
    mask=foreground(rgb);H=components(rgb,he=True)[:,:,0]
    certificate=json.loads(Path(certificate_path).read_text()) if certificate_path else None
    generator_digest=sha256(generator_path);risk_digest=sha256(risk_path) if risk_path else ''
    display=np.full_like(rgb,220);generated=np.zeros_like(rgb);probabilities=[];rows=[]
    for y in range(0,height,w):
        for x in range(0,width,w):
            he=to_tensor(rgb[y:y+w,x:x+w])[None]
            wid=f'{source_id}:R{y//w+1}C{x//w+1}'
            with torch.no_grad():
                fake,var,aux,_=generated_batch(model.netG,he,[wid],rm['samples'] if rm else 3)
                score=float(risk(he,fake,var).sigmoid()) if risk is not None else float('nan')
            valid=bool(mask[y:y+w,x:x+w].mean()>=.5 and H[y:y+w,x:x+w].std()>=.08)
            allowed,reason=gate([score],certificate,generator_digest,risk_digest)
            accepted=bool(valid and allowed[0] and not gm['demo_only'] and rm is not None and not rm['demo_only'])
            if gm['demo_only'] or rm is not None and rm['demo_only']:reason='demo_checkpoint_not_certified'
            crop=to_rgb(fake[0]);generated[y:y+w,x:x+w]=crop
            if accepted:display[y:y+w,x:x+w]=crop
            rows.append(dict(window_id=wid,bbox=[x,y,w,w],quality=valid,accepted=accepted,
                             raw_error_score=score if np.isfinite(score) else None,reason=reason))
            if aux is not None:probabilities.append(aux['probabilities'][0,:,0,0].tolist())
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    save_png(out/'generated_RESEARCH_ONLY.png',Image.fromarray(generated))
    save_png(out/'selective_display.png',Image.fromarray(display))
    write_json(out/'prediction.json',dict(windows=rows,probabilities=probabilities,
        release_coverage=sum(r['accepted'] for r in rows)/len(rows),
        reference_ihc_accessed=False,probability_target='regional DAB proxy bins, not cell HER2 status'))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--threads',type=int,default=1)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--pairs',required=True);p.add_argument('--out',required=True)
    p.add_argument('--window',type=int,default=128);p.add_argument('--auto-split-groups',action='store_true')
    p=sub.add_parser('train');p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model',choices=['cut','expression_cut'],default='expression_cut');p.add_argument('--ngf',type=int,default=64)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--backbone-checkpoint',default='')
    p=sub.add_parser('challenge');p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('fit-risk');p.add_argument('--manifest',required=True);p.add_argument('--generator',required=True)
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=5000)
    p=sub.add_parser('evaluate-generator');p.add_argument('--manifest',required=True);p.add_argument('--generator',required=True)
    p.add_argument('--out',required=True);p.add_argument('--split',choices=['val','test'],default='test')
    for command in ('certify','evaluate'):
        p=sub.add_parser(command);p.add_argument('--manifest',required=True);p.add_argument('--generator',required=True)
        p.add_argument('--risk',required=True);p.add_argument('--out',required=True)
        if command=='certify':p.add_argument('--budget',type=float,default=.1)
        else:p.add_argument('--certificate')
    p=sub.add_parser('infer');p.add_argument('--he',required=True);p.add_argument('--generator',required=True)
    p.add_argument('--risk');p.add_argument('--certificate');p.add_argument('--out',required=True)
    p=sub.add_parser('demo');p.add_argument('--he',required=True);p.add_argument('--ihc',required=True)
    p.add_argument('--out',default='artifacts/attachment');p.add_argument('--steps',type=int,default=120)
    args=parser.parse_args();seed_everything(threads=args.threads)
    if args.command=='prepare':
        pairs=load_pair_csv(args.pairs)
        if args.auto_split_groups:pairs=automatic_group_split(pairs)
        rows=prepare(pairs,args.out,args.window);print('same-coordinate windows',len(rows),'valid',sum(r['valid'] for r in rows))
    elif args.command=='train':
        kwargs={'backbone_checkpoint':args.backbone_checkpoint} if args.model=='expression_cut' else {}
        opt=make_options(args.manifest,args.out,args.model,ngf=args.ngf,batch_size=args.batch_size,**kwargs)
        train_generator(opt,args.steps,args.out)
    elif args.command=='challenge':
        rows=read_manifest(args.manifest);state=fit_morphology([r for r in rows if r['split']=='train'])
        pairs=retrieve([r for r in rows if r['split']=='test'],state)
        write_json(Path(args.out)/'morphology.json',state);write_json(Path(args.out)/'challenge.json',pairs)
    elif args.command=='fit-risk':
        model,opt,gm=load_generator(args.generator);opt.pair_manifest=args.manifest
        dataset=make_dataset(opt,'risk_train')
        if gm['demo_only'] or set(gm['training_groups']) & {r['group_id'] for r in dataset.rows}:
            raise ValueError('Risk training must use groups disjoint from generator training.')
        train_risk(model.netG,dataset,args.out,args.generator,steps=args.steps)
    elif args.command=='evaluate-generator':
        model,opt,gm=load_generator(args.generator);opt.pair_manifest=args.manifest
        ds=make_dataset(opt,args.split)
        if gm['demo_only'] or set(gm['training_groups']) & {r['group_id'] for r in ds.rows}:
            raise ValueError('Held-out generator evaluation cannot use demo or generator-training groups.')
        records,_=evaluate_generator(model.netG,ds,samples=1,keep_outputs=False)
        write_json(Path(args.out)/'generator_evaluation.json',dict(pooled=metric_summary(records),records=records))
    elif args.command in ('certify','evaluate'):
        model,opt,gm=load_generator(args.generator);opt.pair_manifest=args.manifest
        risk,rm=load_risk(args.risk,args.generator)
        if gm['demo_only'] or rm['demo_only']:raise ValueError('Attachment-overfit weights cannot certify or evaluate clinical/generalization risk.')
        training_groups=sorted(set(gm['training_groups'])|set(rm['training_groups']))
        def collect(split):
            ds=make_dataset(opt,split)
            if set(training_groups)&{r['group_id'] for r in ds.rows}:raise ValueError('Evaluation group leaked into training.')
            return evaluate_generator(model.netG,ds,risk,samples=rm['samples'],delta=rm['delta'],keep_outputs=False)[0]
        if args.command=='certify':
            provenance=dict(generator_hash=sha256(args.generator),risk_hash=sha256(args.risk),
                            training_groups=training_groups,delta=rm['delta'],window=rm['window'],samples=rm['samples'],
                            protocol_hash=policy_hash(rm['window']))
            cert=fit_and_certify(collect('calibrate'),collect('certify'),provenance,budget=args.budget)
            write_json(Path(args.out)/'certificate.json',cert)
        else:
            records=collect('test');cert=json.loads(Path(args.certificate).read_text()) if args.certificate else None
            if cert and ({r['group_id'] for r in records}&set(cert['calibration_groups']+cert['certification_groups'])):
                raise ValueError('Test source groups overlap calibration.')
            selected,reason=gate([r['score'] for r in records],cert,sha256(args.generator),sha256(args.risk))
            write_json(Path(args.out)/'test.json',dict(pooled=metric_summary(records),
                equal_group_roi=metric_summary(representative_per_group(records)),
                released_windows=int(selected.sum()),total_windows=sum(r['split']=='test' for r in read_manifest(args.manifest)),policy=reason,records=records))
    elif args.command=='infer':predict_image(args.he,args.generator,args.risk,args.out,args.certificate)
    else:
        from .validation import attachment_demo
        attachment_demo(args.he,args.ihc,args.out,args.steps)


if __name__=='__main__':main()
