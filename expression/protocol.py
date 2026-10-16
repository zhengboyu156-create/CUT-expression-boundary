"""Frozen measurement/sampling contract, bound to risk checkpoints and certificates."""
import hashlib
import json

SAMPLING_PROTOCOL='he_content_seed_v2'


def quality_policy(window=128,min_tissue=.5,min_h_std=.08):
    return dict(version=2,window=window,he_foreground_od=.20,ihc_observability_od=.02,
                min_tissue=min_tissue,min_h_std=min_h_std,dab_threshold=.20,
                reference_denominator='fixed_HE_foreground',mapping='identity')


def policy_hash(window=128,quality=None):
    contract=dict(quality=quality if quality is not None else quality_policy(window),sampling=SAMPLING_PROTOCOL,
                  error_statistic='abs_hard_DAB_area_difference',inference='conditional_mean')
    return hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
