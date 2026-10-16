"""Independent-group calibration and finite-candidate binomial certification.

Estimand: one fixed-seed uniform valid window from each independent source group,
with groups equally weighted. This is not an all-pixel or all-window guarantee.
"""
import numpy as np
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from .protocol import policy_hash


def representative_per_group(rows,seed=0):
    groups={}
    for r in rows:groups.setdefault(r['group_id'],[]).append(r)
    rng=np.random.default_rng(seed)
    return [sorted(groups[g],key=lambda r:r['window_id'])[int(rng.integers(len(groups[g])))] for g in sorted(groups)]


def transform(scores,calibrator):
    return np.interp(scores,calibrator['x'],calibrator['y'])


def fit_and_certify(a,b,provenance,budget=.10,alpha=.05,thresholds=(.02,.05,.1,.2,.4,.6,1.)):
    ga={r['group_id'] for r in a};gb={r['group_id'] for r in b}
    excluded=set(provenance['training_groups'])
    if ga&gb or (ga|gb)&excluded:
        raise ValueError('Training, probability calibration A, and certification B groups must be disjoint.')
    if len(ga)<2 or len(gb)<2:
        raise ValueError('At least two independent groups in each calibration role are required; one image pair cannot certify risk.')
    kinds={r['group_kind'] for r in a+b}
    if len(kinds)!=1 or not kinds<={'patient','slide'}:
        raise ValueError('Verified, consistently defined patient or slide source groups are required.')
    if any(r['split']!='calibrate' for r in a) or any(r['split']!='certify' for r in b):
        raise ValueError('Calibration roles do not match the manifest.')
    if provenance.get('protocol_hash')!=policy_hash(provenance.get('window',128)):
        raise ValueError('Calibration provenance must bind the current sampling/quality protocol.')
    if not 0<budget<1 or not 0<alpha<1:raise ValueError('Invalid risk or significance budget.')
    thresholds=sorted(set(float(t) for t in thresholds))
    if not thresholds or min(thresholds)<0 or max(thresholds)>1:raise ValueError('Invalid fixed thresholds.')
    aa=representative_per_group(a);bb=representative_per_group(b)
    iso=IsotonicRegression(out_of_bounds='clip').fit([r['score'] for r in aa],[r['event'] for r in aa])
    mapping=dict(x=iso.X_thresholds_.tolist(),y=iso.y_thresholds_.tolist())
    score=transform([r['score'] for r in bb],mapping);event=np.array([r['event'] for r in bb])
    candidates=[]
    for t in thresholds:
        selected=score<=t;n=int(selected.sum());e=int(event[selected].sum())
        upper=float(beta.ppf(1-alpha/len(thresholds),e+1,n-e)) if n and e<n else 1.
        candidates.append(dict(threshold=t,n=n,errors=e,upper=upper,coverage=n/len(bb)))
    passed=[c for c in candidates if c['n'] and c['upper']<=budget]
    chosen=max(passed,key=lambda c:(c['coverage'],c['threshold'])) if passed else None
    return dict(version=1,calibrator=mapping,candidates=candidates,chosen=chosen,budget=budget,alpha=alpha,
                calibration_groups=sorted(ga),certification_groups=sorted(gb),group_kind=next(iter(kinds)),
                provenance=provenance,estimand='equal-source-groups; one uniform valid window per group',
                release_enabled=chosen is not None)


def gate(scores,certificate,generator_hash,risk_hash):
    """No IHC input, and no implicit acceptance when calibration is absent/stale."""
    scores=np.asarray(scores,dtype=float)
    if certificate is None:return np.zeros_like(scores,dtype=bool),'missing_certificate'
    p=certificate['provenance']
    if p['generator_hash']!=generator_hash or p['risk_hash']!=risk_hash:
        return np.zeros_like(scores,dtype=bool),'checkpoint_mismatch'
    if p.get('protocol_hash')!=policy_hash(p.get('window',128)):
        return np.zeros_like(scores,dtype=bool),'protocol_mismatch'
    if certificate['chosen'] is None:return np.zeros_like(scores,dtype=bool),'no_certified_threshold'
    chosen=certificate['chosen']
    if not certificate.get('release_enabled') or chosen.get('upper',1)>certificate['budget']:
        return np.zeros_like(scores,dtype=bool),'invalid_certificate_policy'
    valid=np.isfinite(scores)&(scores>=0)&(scores<=1)
    return valid & (transform(scores,certificate['calibrator'])<=chosen['threshold']),'certified_policy' if valid.all() else 'invalid_scores_rejected'


def fixed_coverage_report(rows,coverages=(.5,.8,1.)):
    ordered=sorted(rows,key=lambda r:(r['score'],r['window_id']));result=[]
    for coverage in coverages:
        n=max(1,int(round(len(ordered)*coverage))) if ordered else 0
        selected=ordered[:n];errors=sum(r['event'] for r in selected)
        result.append(dict(coverage=coverage,n=n,errors=errors,risk=errors/n if n else None))
    return result
