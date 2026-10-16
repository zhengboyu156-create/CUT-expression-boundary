"""Run one predeclared multi-pair experiment; supports frozen v1 source on PYTHONPATH."""
import argparse
import gc
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from expression.runner import (seed_everything,make_options,make_dataset,train_generator,
                               evaluate_generator,metric_summary,train_risk,load_generator)
from expression.statistics import StainStatistics,to_rgb


def dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',required=True)
    p.add_argument('--manifests',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=160);a=p.parse_args()
    base=Path(a.manifests).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    seed_everything();opt=make_options(base/'train_manifest.jsonl',out,'expression_cut','demo',ngf=8,batch_size=4)
    if a.variant=='v2':
        opt.learn_bin_means=True;opt.lambda_moment=2.;opt.teacher_probability=.5
    if a.variant=='compat':
        from expression.protocol import SAMPLING_PROTOCOL,policy_hash
        checkpoint=torch.load(base/'v1/generator.pt',map_location='cpu',weights_only=True)
        checkpoint['metadata']['options'].update(learn_bin_means=False,lambda_moment=0.,teacher_probability=1.)
        checkpoint['metadata'].update(sampling_protocol=SAMPLING_PROTOCOL,protocol_hash=policy_hash(),architecture_version=2,
            weights_origin='Copied unchanged from v1 160-step multi-pair run; no third generator training run.')
        path=out/'generator.pt';torch.save(checkpoint,path)
        model,opt,meta=load_generator(path);opt.isTrain=True;dataset=make_dataset(opt)
    else:model,dataset,meta,path=train_generator(opt,a.steps,out)
    records={};summary={};images={};ds={};stat=StainStatistics()
    def add_spatial_metrics(record,output):
        m=output['mask']>0;pred=(stat.dab(output['fake'][None])[0]>.2)&m
        ref=(stat.dab(output['reference'][None])[0]>.2)&m
        union=int((pred|ref).sum());inter=int((pred&ref).sum())
        record['dab_iou']=inter/union if union else None
        record['rgb_l1']=float(((output['fake']-output['reference']).abs()*m).sum()/(3*m.sum().clamp_min(1)))
    for role in ['train','dev','holdout']:
        local=vars(opt).copy();local['pair_manifest']=str(base/(role+'_manifest.jsonl'))
        from types import SimpleNamespace
        ds[role]=make_dataset(SimpleNamespace(**local));by_id={r['window_id']:r for r in ds[role].rows}
        def sink(output,record):
            add_spatial_metrics(record,output)
            row=by_id[output['id']];pid=row['pair_id'];x,y,w,h=row['he_bbox']
            if pid not in images:
                width,height=Image.open(row['he']).size
                images[pid]=np.full((height,width,3),220,dtype=np.uint8)
            images[pid][y:y+h,x:x+w]=to_rgb(output['fake'])
        kwargs=dict(batch_size=4,samples=1)
        if 'keep_outputs' in inspect.signature(evaluate_generator).parameters:
            kwargs.update(keep_outputs=False,output_callback=sink)
        rr,outputs=evaluate_generator(model.netG,ds[role],**kwargs)
        if outputs:
            for r,o in zip(rr,outputs):sink(o,r)
        records[role]=rr
        q=[r['q_reference'] for r in rr]
        summary[role]=dict(metric_summary(rr),zero_fraction_mae=float(np.mean(np.abs(q))),
            bin0_fraction=float(np.mean(np.array(q)<.2)),
            dab_iou=float(np.mean([r['dab_iou'] for r in rr if r['dab_iou'] is not None])),
            rgb_l1=float(np.mean([r['rgb_l1'] for r in rr])))
        dump(out/(role+'_records.json'),rr)
        del outputs;gc.collect()
    # Compare deployment mean conditions to oracle conditions on all held-out windows.
    oracle_errors=[];condition_errors=[];controls=[];net=model.netG.eval()
    with torch.no_grad():
        for batch in DataLoader(ds['holdout'],batch_size=4,shuffle=False):
            q,_=stat(batch['paired_B'],batch['tissue_mask'])
            oracle=net(batch['A'],condition=q);oq,_=stat(oracle,batch['tissue_mask'])
            oracle_errors.extend((oq-q).abs().flatten().tolist())
            _,aux=net(batch['A'],return_aux=True)
            condition_errors.extend((aux['condition']-q).abs().flatten().tolist())
            measured=[]
            for s in [.1,.5,.9]:
                fake=net(batch['A'],condition=torch.full_like(q,s));v,_=stat(fake,batch['tissue_mask']);measured.append(v.flatten().tolist())
            for wid,values in zip(batch['window_id'],zip(*measured)):
                controls.append(dict(window_id=wid,measured=list(values)))
    summary['conditioning']=dict(oracle_stat_mae=float(np.mean(oracle_errors)),
        deployed_condition_mae=float(np.mean(condition_errors)),mean_high_minus_low=float(np.mean([c['measured'][-1]-c['measured'][0] for c in controls])),
        monotone_fraction=float(np.mean([c['measured'][0]<=c['measured'][1]<=c['measured'][2] for c in controls])))
    dump(out/'controls.json',controls)
    # h is trained only on the development image-pair, never the G train/holdout images.
    # All roles are explicitly same-source demo: this is not independent calibration.
    if a.variant in ('v2','compat'):
        risk,rm,rp=train_risk(model.netG,ds['dev'],out,path,steps=160,batch_size=4,samples=3,cache_demo=True)
        rr,_=evaluate_generator(model.netG,ds['holdout'],risk,batch_size=4,samples=3,keep_outputs=False)
        summary['raw_risk_holdout']=metric_summary(rr)
        events=np.array([r['event'] for r in rr]);scores=np.array([r['score'] for r in rr])
        from sklearn.metrics import roc_auc_score
        summary['raw_risk_holdout'].update(auroc=float(roc_auc_score(events,scores)) if len(np.unique(events))==2 else None,
                                          event_classes=np.unique(events).tolist(),brier=float(np.mean((scores-events)**2)),
                                          status='within-source image-pair holdout; uncalibrated')
        summary['risk_training_events']=rm.get('observed_error_events')
        dump(out/'holdout_risk_records.json',rr)
    # Buffered + verified export avoids silently delivering an incomplete final PNG.
    from expression.artifact_io import save_png
    for pid,img in images.items():save_png(out/(pid+'_generated.png'),Image.fromarray(img))
    summary.update(variant=a.variant,steps=a.steps,ngf=8,residual_blocks=6,
        scope='image-pair debugging only; 2M source shared across roles; no patient-level validation',
        dataset_cache=dict(hits=getattr(dataset,'cache_hits',None),misses=getattr(dataset,'cache_misses',None)))
    if a.variant=='compat':summary['weights_origin']=meta['weights_origin']
    dump(out/'summary.json',summary);print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
