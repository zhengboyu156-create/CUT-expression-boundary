"""Recompute evidence from actual attachments/weights; do not manufacture research success.

Run from repository root: python -m tools.audit_complete_evidence
All selection is deterministic. This audits frozen debug weights; it does not
train a new G or turn unknown image sources into independent patients.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score,roc_auc_score
from expression.manifest import read_manifest,sha256
from expression.statistics import foreground,components,to_rgb
from expression.runner import seed_everything,load_generator,make_dataset,generated_batch,load_risk
from expression.challenge import fit_morphology,retrieve
from expression.calibration import gate,fit_and_certify
from expression.protocol import policy_hash
from expression.artifact_io import save_png

ROOT=Path(__file__).resolve().parents[1]


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def independent_dab(image):
    """NumPy float64, literal fixed stain vectors; no StainStatistics call."""
    h=np.array([.65,.70,.29],dtype=np.float64);h/=np.linalg.norm(h)
    d=np.array([.27,.57,.78],dtype=np.float64);d/=np.linalg.norm(d)
    third=np.cross(h,d);third/=np.linalg.norm(third)
    inverse=np.linalg.inv(np.column_stack((h,d,third)))
    rgb=np.clip((np.asarray(image,dtype=np.float64)+1)*127.5,0,255)
    return np.maximum((-np.log((rgb+1)/256))@inverse[1],0)


def region_metrics(fake,reference,mask):
    a=independent_dab(fake);b=independent_dab(reference)
    pred=(a>.2)&mask;truth=(b>.2)&mask
    union=int((pred|truth).sum());intersection=int((pred&truth).sum())
    q_pred=float(pred.sum()/mask.sum());q_ref=float(truth.sum()/mask.sum())
    return dict(q_reference=q_ref,q_generated=q_pred,error=abs(q_pred-q_ref),
        event=int(abs(q_pred-q_ref)>.1),dab_iou=intersection/union if union else None,
        rgb_l1=float(np.abs(fake-reference)[mask].mean()),
        intersection=intersection,union=union,reference_positive=int(truth.sum()),
        generated_positive=int(pred.sum()),mask_pixels=int(mask.sum()))


def run(base,out):
    seed_everything();out.mkdir(parents=True,exist_ok=True)
    rows=read_manifest(base/'full_manifest.jsonl');valid=[r for r in rows if r['valid']]
    train=[r for r in valid if r['debug_partition']=='train']
    unseen=[r for r in valid if r['debug_partition']!='train']
    original=[json.loads(x) for x in (base/'v1_full_manifest.jsonl').read_text().splitlines()]
    identity=[];measurement=[];hashes={};old_disagreements=0
    for pid in sorted({r['pair_id'] for r in rows}):
        rr=[r for r in rows if r['pair_id']==pid];r=rr[0]
        he=np.array(Image.open(r['he']).convert('RGB'));ihc=np.array(Image.open(r['ihc']).convert('RGB'))
        hm=foreground(he);im_old=foreground(ihc);H=components(he,he=True)[:,:,0]
        coverage=np.zeros(he.shape[:2],dtype=np.uint8)
        for x in rr:
            a,b,w,h=x['he_bbox'];coverage[b:b+h,a:a+w]+=1
            assert x['he_bbox']==x['ihc_bbox']
            mask=hm[b:b+h,a:a+w]
            raw=ihc[b:b+h,a:a+w].astype(np.float64)/127.5-1
            q=float(((independent_dab(raw)>.2)&mask).sum()/mask.sum()) if mask.any() else None
            measurement.append(dict(window_id=x['window_id'],manifest_q=x['q'],independent_q=q,
                absolute_difference=abs(q-x['q']) if q is not None else None))
            legacy=mask.mean()>=.5 and H[b:b+h,a:a+w].std()>=.08 and im_old[b:b+h,a:a+w].mean()>=.5
            oldrow=next(t for t in original if t['window_id']==x['window_id'])
            old_disagreements+=int(bool(legacy)!=oldrow['valid'])
        assert np.all(coverage==1)
        identity.append(dict(pair_id=pid,windows=len(rr),covered_once_pixels=int((coverage==1).sum()),
            total_pixels=int(coverage.size),valid=sum(x['valid'] for x in rr)))
        for k in ['he','ihc']:hashes[str(Path(r[k]).relative_to(ROOT))]=sha256(r[k])
    dump(out/'input_identity.json',dict(pairs=identity,all_448_coordinates_identical=True,
        image_hashes=hashes,old_qc_reproduction_disagreements=old_disagreements,
        max_reference_measurement_difference=max(x['absolute_difference'] or 0 for x in measurement)))
    dump(out/'independent_reference_measurements.json',measurement)
    assert old_disagreements==0
    # Train-only fit, frozen radius and expression thresholds; all comparisons
    # are between H&E descriptors. Each item retains its own IHC coordinates.
    state=fit_morphology(train,demo=True);all_pairs=retrieve(valid,state,demo=True)
    unseen_pairs=retrieve(unseen,state,demo=True)
    by_id={r['window_id']:r for r in valid}
    z=(np.array([r['features'] for r in valid])-state['mean'])/state['scale']
    centers=np.array(state['centers']);labels=((z[:,None]-centers)**2).sum(2).argmin(1)
    distances=np.sqrt(((z[:,None]-z[None])**2).sum(-1));q=np.array([r['q'] for r in valid])
    eligible=(distances<=1.5)&(np.abs(q[:,None]-q[None])>=.2)&(labels[:,None]==labels[None])&np.triu(np.ones(distances.shape,dtype=bool),1)
    expected={(valid[i]['window_id'],valid[j]['window_id']) for i,j in zip(*np.where(eligible))}
    actual={tuple(sorted((p['a'],p['b']))) for p in all_pairs}
    assert {tuple(sorted(x)) for x in expected}==actual
    # Reproduce the former top-16 cap on identical descriptors, without changing inputs.
    from sklearn.neighbors import NearestNeighbors
    dd,jj=NearestNeighbors(n_neighbors=17).fit(z).kneighbors(z);oldset=set()
    for i,(ds,js) in enumerate(zip(dd,jj)):
        for d,j in zip(ds,js):
            if i!=j and labels[i]==labels[j] and d<=1.5 and abs(q[i]-q[j])>=.2:
                oldset.add(tuple(sorted((valid[i]['window_id'],valid[j]['window_id']))))
    pair_sources=Counter()
    for p in all_pairs:
        a,b=by_id[p['a']],by_id[p['b']]
        pair_sources['same_pair' if a['pair_id']==b['pair_id'] else 'different_pairs']+=1
    challenge=dict(fit_windows=len(train),unseen_windows=len(unseen),frozen_distance_threshold=1.5,
        frozen_delta_threshold=.2,old_top16_candidates=len(oldset),complete_candidates=len(all_pairs),
        exhaustive_count=int(eligible.sum()),missing_candidates_before=len(actual-oldset),
        unseen_image_pair_candidates=len(unseen_pairs),unseen_reference_q_min=min(r['q'] for r in unseen),
        unseen_reference_q_max=max(r['q'] for r in unseen),candidate_sources=dict(pair_sources),
        actual_candidate_group_ids=sorted({by_id[p[k]]['group_id'] for p in all_pairs for k in ['a','b']}),
        independent_patient_challenge_validated=False,
        interpretation='All candidates are staining-proxy diagnostics; no verified protein-opposite patient challenge.')
    dump(out/'challenge_audit.json',challenge);dump(out/'morphology_train_only.json',state)
    dump(out/'complete_challenge_candidates.json',all_pairs);dump(out/'unseen_challenge_candidates.json',unseen_pairs)
    print('INPUT_AND_CHALLENGE',json.dumps(challenge),flush=True)
    # Independently measure floating-point neural outputs from the actual weights.
    train_counts=np.bincount([min(int(r['q']*5),4) for r in train],minlength=5)
    prior=(train_counts+1)/(len(train)+5);models={};records_all={};selected={}
    training_pixel_prevalence=sum(r['q']*r['he_tissue'] for r in train)/sum(r['he_tissue'] for r in train)
    dump(out/'training_pixel_baseline.json',dict(training_windows=len(train),
         training_foreground_pixels=round(sum(r['he_tissue']*16384 for r in train)),
         training_positive_pixels=round(sum(r['q']*r['he_tissue']*16384 for r in train)),
         training_pixel_prevalence=training_pixel_prevalence,
         selection='All valid training-role windows only; no dev/holdout targets used.'))
    hold_rows=[r for r in valid if r['debug_partition']=='holdout']
    selected_id=max(hold_rows,key=lambda r:(r['q'],r['window_id']))['window_id']
    for variant in ['v2_compat','v2']:
        path=base/variant/'generator.pt';model,opt,meta=load_generator(path)
        opt.pair_manifest=str(base/'holdout_manifest.jsonl');ds=make_dataset(opt)
        risk,rm=load_risk(base/variant/'risk.pt',path)
        rr=[];controls=[];pixels=[];truths=[];distributions_invariant=True;max_reproduction_difference=0.
        old={r['window_id']:r for r in json.loads((base/variant/'holdout_records.json').read_text())}
        for batch in DataLoader(ds,batch_size=4,shuffle=False):
            with torch.no_grad():
                fake,variance,aux,draws=generated_batch(model.netG,batch['A'],batch['window_id'],samples=3)
                score=risk(batch['A'],fake,variance).sigmoid().flatten().tolist()
                sweep={};lo_aux=None;hi_aux=None
                for condition in [0.,.1,.3,.5,.7,.9,1.]:
                    image,condition_aux=model.netG(batch['A'],condition=torch.full((len(fake),1,1,1),condition),return_aux=True)
                    sweep[condition]=image.detach().cpu().numpy().transpose(0,2,3,1)
                    if condition==0.:lo_aux=condition_aux
                    if condition==1.:hi_aux=condition_aux
                for key in ['logits','pixel_logits','expected_condition']:
                    distributions_invariant &= torch.equal(lo_aux[key],hi_aux[key])
            for i,wid in enumerate(batch['window_id']):
                a=fake[i].cpu().numpy().transpose(1,2,0);b=batch['paired_B'][i].numpy().transpose(1,2,0)
                mask=batch['tissue_mask'][i,0].numpy()>0
                record=region_metrics(a,b,mask);record.update(window_id=wid,score=score[i])
                probs=aux['probabilities'][i,:,0,0].cpu().numpy().astype(float)
                label=min(int(record['q_reference']*5),4)
                record.update(probabilities=probs.tolist(),bin_target=label,
                    bin_log_loss=float(-np.log(probs[label])),prior_log_loss=float(-np.log(prior[label])),
                    predicted_condition=float(aux['expected_condition'][i]),
                    zero_fraction_error=record['q_reference'])
                pixel_prob=aux['pixel_logits'][i,0].sigmoid().cpu().numpy();pixel_truth=independent_dab(b)[mask]>.2
                pixels.append(pixel_prob[mask]);truths.append(pixel_truth)
                reference_by_numpy=record['q_reference']
                measured=[]
                for condition,image in sweep.items():
                    q_cond=float(((independent_dab(image[i])>.2)&mask).sum()/mask.sum())
                    measured.append(dict(condition=condition,measured_q=q_cond))
                controls.append(dict(window_id=wid,reference_q=reference_by_numpy,points=measured))
                for key in ['q_reference','q_generated','error','dab_iou','rgb_l1']:
                    if old[wid][key] is not None:
                        max_reproduction_difference=max(max_reproduction_difference,abs(record[key]-old[wid][key]))
                rr.append(record)
                if wid==selected_id:
                    sample_q=[float(((independent_dab(draw.numpy().transpose(1,2,0))>.2)&mask).sum()/mask.sum()) for draw in draws[i].cpu()]
                    selected[variant]=dict(window_id=wid,bbox=by_id[wid]['he_bbox'],reference_q=reference_by_numpy,
                        predicted_condition=record['predicted_condition'],sample_generated_q=sample_q,
                        sample_rgb_variance_mean=float(variance[i].mean()),probabilities=probs.tolist())
                    for name,img in [('he',batch['A'][i]),('reference',batch['paired_B'][i]),('generated',fake[i])]:
                        save_png(out/f'{variant}_{name}_crop.png',Image.fromarray(to_rgb(img)))
                    np.savez_compressed(out/f'{variant}_pixel_evidence.npz',pixel_probability=pixel_prob,
                        reference_positive=independent_dab(b)>.2,generated_positive=independent_dab(a)>.2,mask=mask)
                    for j,draw in enumerate(draws[i]):save_png(out/f'{variant}_sample_{j}.png',Image.fromarray(to_rgb(draw)))
        p=np.concatenate(pixels).astype(float);t=np.concatenate(truths).astype(float)
        clipped=np.clip(p,1e-7,1-1e-7);prevalence=float(t.mean())
        event=np.array([r['event'] for r in rr]);scores=np.array([r['score'] for r in rr])
        # Spatial failure is an additional diagnostic, not a retrospectively
        # relabelled training objective or a clinically validated error event.
        spatial=[int(r['dab_iou'] is not None and r['dab_iou']<.1) for r in rr]
        coverage=[]
        order=sorted(rr,key=lambda r:(r['score'],r['window_id']))
        for cover in [.5,.8,1.]:
            n=int(round(len(order)*cover));sel=order[:n]
            coverage.append(dict(coverage=cover,n=n,area_error_rate=float(np.mean([r['event'] for r in sel])),
                localization_failure_rate=float(np.mean([r['dab_iou'] is not None and r['dab_iou']<.1 for r in sel]))))
        m=dict(n=len(rr),area_mae=float(np.mean([r['error'] for r in rr])),zero_baseline_mae=float(np.mean([r['q_reference'] for r in rr])),
            mean_dab_iou=float(np.mean([r['dab_iou'] for r in rr if r['dab_iou'] is not None])),rgb_l1=float(np.mean([r['rgb_l1'] for r in rr])),
            bin_log_loss=float(np.mean([r['bin_log_loss'] for r in rr])),train_histogram_log_loss=float(np.mean([r['prior_log_loss'] for r in rr])),
            bin_accuracy=float(np.mean([np.argmax(r['probabilities'])==r['bin_target'] for r in rr])),
            reference_bin_counts=dict(Counter(str(r['bin_target']) for r in rr)),
            pixel_dab_average_precision=float(average_precision_score(t,p)),pixel_positive_prevalence=prevalence,
            pixel_bce=float(np.mean(-t*np.log(clipped)-(1-t)*np.log(1-clipped))),
            pixel_train_prevalence=training_pixel_prevalence,
            pixel_train_prevalence_bce=float(-prevalence*np.log(training_pixel_prevalence)-(1-prevalence)*np.log(1-training_pixel_prevalence)),
            pixel_constant_prevalence_bce=float(-prevalence*np.log(prevalence)-(1-prevalence)*np.log(1-prevalence)),
            high_low_q_response=float(np.mean([c['points'][-2]['measured_q']-c['points'][1]['measured_q'] for c in controls])),
            mean_condition_realization_error=float(np.mean([abs(p['measured_q']-p['condition']) for c in controls for p in c['points']])),
            teacher_cannot_change_HE_distribution=bool(distributions_invariant),
            risk_event_counts=dict(Counter(str(int(e)) for e in event)),risk_training_event_counts=rm['observed_error_events'],
            raw_risk_auroc=float(roc_auc_score(event,scores)) if len(np.unique(event))==2 else None,
            raw_risk_brier=float(np.mean((scores-event)**2)),fixed_coverage=coverage,
            area_event_zero_but_spatial_diagnostic_failed=sum(r['event']==0 and r['dab_iou'] is not None and r['dab_iou']<.1 for r in rr),
            max_independent_metric_difference_from_saved=max_reproduction_difference,
            generator_sha256=sha256(path),risk_sha256=sha256(base/variant/'risk.pt'),
            source='Saved 160-step ngf=8 frozen debug weights; no new G training.')
        assert max_reproduction_difference<1e-6,(variant,max_reproduction_difference)
        assert distributions_invariant
        dump(out/f'{variant}_independent_holdout_records.json',rr);dump(out/f'{variant}_condition_sweep.json',controls)
        records_all[variant]=rr;models[variant]=m
        print('MODEL',variant,json.dumps(m),flush=True)
    dump(out/'selected_sampling_evidence.json',selected)
    # Exact G equality after the earlier compatibility migration.
    v1=torch.load(base/'v1/generator.pt',map_location='cpu',weights_only=True)
    compat=torch.load(base/'v2_compat/generator.pt',map_location='cpu',weights_only=True)
    weight_difference=max(float((v1['generator'][k]-compat['generator'][k]).abs().max()) for k in v1['generator'])
    assert weight_difference==0.
    traces={}
    for variant in ['v1','v2']:
        log=json.loads((base/variant/'training.json').read_text())
        finite=all(np.isfinite(v) for r in log for v in r.values());assert finite and len(log)==160
        traces[variant]=dict(steps=len(log),sequential_steps=[r['step'] for r in log]==list(range(1,161)),all_losses_finite=finite,
            training_log_sha256=sha256(base/variant/'training.json'),checkpoint_sha256=sha256(base/variant/'generator.pt'))
    # Instrument the real deployment entry point. Any file-image read except the
    # supplied H&E is rejected; PNG decode checks on in-memory buffers are allowed.
    from expression import cli
    he_path=by_id[selected_id]['he'];opened=[];export_reads=[];hash_calls=[];original_open=Image.open;original_hash=cli.sha256
    def guarded_open(fp,*args,**kwargs):
        if isinstance(fp,(str,Path)):
            resolved=Path(fp).resolve()
            if resolved==Path(he_path).resolve():opened.append(str(resolved))
            elif resolved.parent==(out/'he_only_inference').resolve():export_reads.append(str(resolved))
            else:raise AssertionError('Unexpected file-image read '+str(resolved))
        return original_open(fp,*args,**kwargs)
    def counted_hash(path):hash_calls.append(str(Path(path).resolve()));return original_hash(path)
    with patch.object(Image,'open',guarded_open),patch.object(cli,'sha256',counted_hash):
        inference=cli.predict_image(he_path,base/'v2_compat/generator.pt',base/'v2_compat/risk.pt',out/'he_only_inference')
    expected_hash_paths=[str((base/'v2_compat/generator.pt').resolve()),str((base/'v2_compat/risk.pt').resolve())]
    assert hash_calls==expected_hash_paths
    assert all(not r['accepted'] for r in inference)
    inference_audit=dict(input_file_image_reads=opened,output_png_decode_checks=export_reads,
        reference_ihc_accessed=False,window_count=len(inference),
        released_windows=sum(r['accepted'] for r in inference),gate_digest_calls=hash_calls,
        digest_calls_per_file=dict(Counter(hash_calls)),reason_counts=dict(Counter(r['reason'] for r in inference)),
        scope='Actual H&E-only entry point with real debug G/h; no certificate exists.')
    dump(out/'he_only_access_audit.json',inference_audit)
    # Test actual-source overlap without assigning invented patient identities.
    a=[dict(r,score=.1,event=0,split='calibrate') for r in valid if r['debug_partition']=='dev']
    b=[dict(r,score=.1,event=0,split='certify') for r in valid if r['debug_partition']=='holdout']
    try:
        fit_and_certify(a,b,dict(training_groups=[],generator_hash='g',risk_hash='h',protocol_hash=policy_hash()))
        raise AssertionError('Actual overlapping sources were certified.')
    except ValueError as exc:actual_rejection=str(exc)
    K=7;alpha=.05;budget=.1
    certification=dict(actual_source_attempt_rejected=actual_rejection,
        actual_distinct_source_ids=len({r['group_id'] for r in valid}),verified_patient_groups=0,
        candidate_threshold_count=K,alpha=alpha,risk_budget=budget,
        zero_errors_upper=[dict(selected_independent_groups=n,upper=1-(alpha/K)**(1/n)) for n in [1,2,10,46,47,100]],
        minimum_accepted_independent_B_groups_if_zero_errors=math.ceil(math.log(alpha/K)/math.log(1-budget)),
        formula='U(e=0,n)=1-(alpha/K)^(1/n); general U=BetaQuantile(1-alpha/K,e+1,n-e)',
        guarantee_assumptions=['G, h, error definition and candidate thresholds frozen before B',
            'A, B, training and test source groups independent and disjoint',
            'one uniformly sampled valid window per independent group; groups equally weighted',
            'deployment shares the certified source distribution and quality protocol'],
        actual_certificate_exists=False,actual_trusted_release_coverage=0.)
    dump(out/'certification_audit.json',certification)
    max_input_diff=max(x['absolute_difference'] or 0 for x in measurement)
    result=dict(scope='Evidence audit of actual attachments, NOT a positive clinical validation',
        input_pairs=7,new_pairs=6,total_raw_windows=len(rows),valid_windows=len(valid),
        all_coordinates_identity=True,max_independent_reference_difference=max_input_diff,
        challenge=challenge,models=models,compat_generator_weight_difference=weight_difference,
        training_trace_checks=traces,inference=inference_audit,certification=certification,
        no_new_generator_training=True,
        conclusions=dict(marker_specific_predictability='not_established',
            biological_expression_posterior='not_established; measured distribution and localization fail controls',
            calibrated_pathology_risk='not_established; no independent certification data',
            software_identity_and_fail_closed_deployment='executed_and_verified'))
    dump(out/'complete_evidence.json',result)
    # Flat, independently recomputed per-window evidence, easy to inspect elsewhere.
    import csv
    with (out/'independent_holdout_metrics.csv').open('w',newline='') as f:
        columns=['variant','window_id','q_reference','q_generated','error','event','dab_iou','rgb_l1','score','bin_log_loss','prior_log_loss']
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for variant,rr in records_all.items():
            for r in rr:w.writerow(dict(variant=variant,**{k:r[k] for k in columns if k!='variant'}))
    print('EVIDENCE_COMPLETE',out,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='artifacts/multipair_debug');parser.add_argument('--out',default='artifacts/complete_evidence')
    args=parser.parse_args();run(Path(args.base).resolve(),Path(args.out).resolve())


if __name__=='__main__':main()
