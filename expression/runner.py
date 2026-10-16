"""Headless training and evaluation using the uploaded CUT implementations."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
from models.cut_model import CUTModel
from models.expression_cut_model import ExpressionCUTModel
from data.expression_pair_dataset import ExpressionPairDataset
from .networks import ConditionalGenerator, ErrorPredictor
from .statistics import StainStatistics
from .manifest import sha256
from .calibration import fixed_coverage_report, gate, representative_per_group
from .manifest import read_manifest
from .protocol import SAMPLING_PROTOCOL,policy_hash


def seed_everything(seed=0,threads=1):
    np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    torch.backends.nnpack.set_flags(False)


def make_options(manifest,out,model='expression_cut',split='train',ngf=64,batch_size=1,**kwargs):
    parser=TrainOptions().initialize(argparse.ArgumentParser(allow_abbrev=False))
    cls=ExpressionCUTModel if model=='expression_cut' else CUTModel
    parser=cls.modify_commandline_options(parser,True,command_line=[])
    if model=='cut':parser.add_argument('--stats_window',type=int,default=128)
    parser=ExpressionPairDataset.modify_commandline_options(parser,True)
    # Both baselines use the same identity-NCE setting for the attachment comparison.
    args=['--pair_manifest',str(manifest),'--pair_split',split,'--netG','resnet_6blocks',
          '--model',model,'--dataset_mode','expression_pair','--gpu_ids','-1',
          '--ngf',str(ngf),'--ndf',str(ngf),'--batch_size',str(batch_size),
          '--nce_idt','false','--no_flip','--preprocess','none','--num_threads','0',
          '--name',model,'--checkpoints_dir',str(out),'--netF_nc','64','--num_patches','32']
    if split=='demo':args+=['--demo_overfit']
    opt=parser.parse_args(args);opt.gpu_ids=[];opt.isTrain=True
    windows={tuple(r['he_bbox'][2:]) for r in read_manifest(manifest)}
    if len(windows)!=1 or next(iter(windows))[0]!=next(iter(windows))[1]:raise ValueError('Manifest must have one square statistic-window size.')
    opt.stats_window=next(iter(windows))[0]
    opt.n_epochs=100000;opt.n_epochs_decay=0
    for k,v in kwargs.items():
        if not hasattr(opt,k):raise ValueError('Unknown model option '+k)
        setattr(opt,k,v)
    return opt


def make_dataset(opt,split=None,training=False):
    d=vars(opt).copy();d['isTrain']=training
    if split is not None:d['pair_split']=split
    return ExpressionPairDataset(SimpleNamespace(**d))


def train_generator(opt,steps,out):
    if steps<1:raise ValueError('Training steps must be positive.')
    dataset=make_dataset(opt,training=True)
    loader=DataLoader(dataset,batch_size=opt.batch_size,shuffle=True,drop_last=True,num_workers=0,
                      generator=torch.Generator().manual_seed(0))
    if len(loader)==0:raise ValueError('Batch exceeds valid dataset size.')
    model=(ExpressionCUTModel if opt.model=='expression_cut' else CUTModel)(opt)
    torch.manual_seed(0);iterator=iter(loader);first=next(iterator)
    model.data_dependent_initialize(first)
    # data_dependent_initialize deliberately creates F before its optimizer.
    log=[]
    for step in range(steps):
        torch.manual_seed(1337+step);np.random.seed(1337+step)
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        model.set_input(batch)
        if hasattr(model,'set_training_progress'):model.set_training_progress(step/max(steps-1,1))
        model.optimize_parameters()
        losses={k:float(v) for k,v in model.get_current_losses().items()}
        if not all(np.isfinite(v) for v in losses.values()):raise FloatingPointError('Non-finite training loss.')
        log.append(dict(step=step+1,**losses))
        if (step+1)%20==0:print(opt.model,'step',step+1,'loss',round(losses['G'],5),flush=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/'generator.pt'
    metadata=dict(options=vars(opt),training_groups=sorted({r['group_id'] for r in dataset.rows}),
                  demo_only=opt.pair_split=='demo',steps=steps,manifest_hash=sha256(opt.pair_manifest),
                  architecture_version=2,sampling_protocol=SAMPLING_PROTOCOL,protocol_hash=policy_hash(opt.stats_window))
    torch.save(dict(generator=model.netG.state_dict(),feature_projector=model.netF.state_dict(),
                    discriminator=model.netD.state_dict(),metadata=metadata),path)
    (out/'training.json').write_text(json.dumps(log,indent=2))
    return model,dataset,metadata,path


def load_generator(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    metadata=checkpoint['metadata'];opt=SimpleNamespace(**metadata['options']);opt.isTrain=False;opt.gpu_ids=[]
    if hasattr(opt,'backbone_checkpoint'):opt.backbone_checkpoint=''
    if opt.model=='expression_cut':opt.learn_bin_means=any('within_bin_head.' in k for k in checkpoint['generator'])
    model=(ExpressionCUTModel if opt.model=='expression_cut' else CUTModel)(opt)
    model.netG.load_state_dict(checkpoint['generator']);model.netG.eval()
    return model,opt,metadata


@torch.no_grad()
def generated_batch(net,he,ids,samples=3):
    if isinstance(net,ConditionalGenerator):
        generated,aux=net(he,return_aux=True)
        draws=[]
        for image,wid in zip(he,ids):
            # IDs are bookkeeping, never a source of stochastic risk differences.
            content=image.detach().cpu().contiguous().numpy().tobytes()
            seed=int(hashlib.sha256(content).hexdigest()[:16],16)%(2**63-1)
            rng=torch.Generator(device=he.device).manual_seed(seed)
            draws.append(torch.stack([net(image[None],sample=True,generator=rng)[0] for _ in range(samples)]))
        draws=torch.stack(draws)
        variance=draws.var(1,unbiased=False).mean(1,keepdim=True)
        return generated,variance,aux,draws
    generated=net(he)
    return generated,torch.zeros_like(he[:,:1]),None,generated[:,None].expand(-1,samples,-1,-1,-1)


@torch.no_grad()
def evaluate_generator(net,dataset,risk=None,batch_size=4,samples=3,delta=.10,keep_outputs=True,output_callback=None):
    net.eval()
    if risk is not None:risk.eval()
    measurement=StainStatistics(dataset.opt.stats_window)
    rows={r['window_id']:r for r in dataset.rows};records=[];outputs=[]
    for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0):
        fake,variance,aux,draws=generated_batch(net,batch['A'],batch['window_id'],samples)
        target,_=measurement(batch['paired_B'],batch['tissue_mask'])
        predicted,_=measurement(fake,batch['tissue_mask'])
        score=risk(batch['A'],fake,variance).sigmoid().flatten() if risk is not None else torch.full((len(fake),),float('nan'))
        for i,wid in enumerate(batch['window_id']):
            row=rows[wid];error=abs(float(predicted[i])-float(target[i]))
            record={k:row[k] for k in ['window_id','group_id','group_kind','split']}
            record.update(q_reference=float(target[i]),q_generated=float(predicted[i]),error=error,
                          event=int(error>delta),score=float(score[i]) if risk is not None else None)
            m=batch['tissue_mask'][i]>0
            pd=(measurement.dab(fake[i:i+1])[0]>.20)&m
            rd=(measurement.dab(batch['paired_B'][i:i+1])[0]>.20)&m
            union=int((pd|rd).sum());intersection=int((pd&rd).sum())
            record['dab_iou']=intersection/union if union else None
            record['rgb_l1']=float(((fake[i]-batch['paired_B'][i]).abs()*m).sum()/(3*m.sum().clamp_min(1)))
            if aux is not None:
                probs=aux['probabilities'][i,:,0,0].cpu().numpy();bin_id=min(int(float(target[i])*5),4)
                record.update(probabilities=probs.tolist(),bin_target=bin_id,
                              log_loss=float(-np.log(max(probs[bin_id],1e-12))))
                record['predicted_condition']=float(aux['expected_condition'][i])
            records.append(record)
            output=dict(id=wid,he=batch['A'][i],reference=batch['paired_B'][i],mask=batch['tissue_mask'][i],
                                fake=fake[i],variance=variance[i],samples=draws[i],
                                pixel_probability=None if aux is None else aux['pixel_logits'][i].sigmoid())
            if keep_outputs:outputs.append(output)
            if output_callback is not None:output_callback(output,record)
    return records,outputs


def train_risk(net,dataset,out,generator_path,steps=150,batch_size=4,samples=3,delta=.10,cache_demo=False):
    if steps<1 or samples<2:raise ValueError('Risk training needs positive steps and at least two generated samples.')
    if dataset.opt.pair_split not in ('risk_train','demo'):raise ValueError('Error predictor needs its own risk_train groups.')
    net.eval();head=ErrorPredictor(dataset.opt.stats_window);optimizer=torch.optim.Adam(head.parameters(),lr=.002)
    measurement=StainStatistics(dataset.opt.stats_window)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    observed_labels={}
    def inputs(batch):
        with torch.no_grad():
            fake,var,_,_=generated_batch(net,batch['A'],batch['window_id'],samples)
            q,_=measurement(fake,batch['tissue_mask']);real,_=measurement(batch['paired_B'],batch['tissue_mask'])
            event=((q-real).abs()>delta).float()
            for wid,label in zip(batch['window_id'],event.flatten().tolist()):observed_labels[wid]=int(label)
        return batch['A'],fake,var,event
    if cache_demo:
        if dataset.opt.pair_split!='demo':raise ValueError('In-memory cache is limited to the small attachment demo.')
        cached=[inputs(b) for b in DataLoader(dataset,batch_size=batch_size,shuffle=False)]
        iterator=None
    else:iterator=iter(loader)
    losses=[]
    for step in range(steps):
        if cache_demo:he,fake,var,event=cached[step%len(cached)]
        else:
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            he,fake,var,event=inputs(batch)
        optimizer.zero_grad();logits=head(he,fake,var);loss=F.binary_cross_entropy_with_logits(logits,event)
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite error-predictor loss.')
        loss.backward();optimizer.step();losses.append(float(loss.detach()))
    head.eval();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metadata=dict(generator_hash=sha256(generator_path),training_groups=sorted({r['group_id'] for r in dataset.rows}),
                  delta=delta,samples=samples,window=dataset.opt.stats_window,demo_only=dataset.opt.pair_split=='demo',
                  sampling_protocol=SAMPLING_PROTOCOL,protocol_hash=policy_hash(dataset.opt.stats_window))
    metadata.update(observed_windows=len(observed_labels),observed_error_events={str(v):sum(x==v for x in observed_labels.values()) for v in [0,1]},
                    discrimination_training_has_both_classes=len(set(observed_labels.values()))==2)
    torch.save(dict(state=head.state_dict(),metadata=metadata),out/'risk.pt')
    (out/'risk_training.json').write_text(json.dumps(losses))
    return head,metadata,out/'risk.pt'


def load_risk(path,generator_path):
    saved=torch.load(path,map_location='cpu',weights_only=True);metadata=saved['metadata']
    if metadata['generator_hash']!=sha256(generator_path):raise ValueError('Risk predictor belongs to a different frozen generator.')
    if metadata.get('sampling_protocol')!=SAMPLING_PROTOCOL or metadata.get('protocol_hash')!=policy_hash(metadata['window']):
        raise ValueError('Risk checkpoint uses an incompatible sampling/quality protocol; retrain it.')
    head=ErrorPredictor(metadata['window']);head.load_state_dict(saved['state']);head.eval()
    return head,metadata


def metric_summary(records):
    if not records:return dict(n=0)
    summary=dict(n=len(records),mae=float(np.mean([r['error'] for r in records])),
                 error_rate=float(np.mean([r['event'] for r in records])))
    if 'log_loss' in records[0]:
        summary['bin_log_loss']=float(np.mean([r['log_loss'] for r in records]))
        summary['bin_accuracy']=float(np.mean([np.argmax(r['probabilities'])==r['bin_target'] for r in records]))
    if 'dab_iou' in records[0]:
        ious=[r['dab_iou'] for r in records if r['dab_iou'] is not None]
        summary.update(mean_dab_iou=float(np.mean(ious)) if ious else None,
                       rgb_l1=float(np.mean([r['rgb_l1'] for r in records])))
    if records[0]['score'] is not None:
        summary['fixed_coverage']=fixed_coverage_report(records)
    return summary
