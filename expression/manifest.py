"""Full-image identity-coordinate tiling and provenance checks."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .statistics import components, descriptor, foreground
from .protocol import quality_policy,policy_hash

SPLITS = ('train', 'val', 'risk_train', 'calibrate', 'certify', 'test', 'demo')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    hashes={}
    for r in rows:
        for key in ('he', 'ihc'):
            p = Path(r[key]); r[key] = str((path.parent / p).resolve()) if not p.is_absolute() else str(p)
            if key+'_sha256' in r:
                if r[key] not in hashes:hashes[r[key]]=sha256(r[key])
                if hashes[r[key]]!=r[key+'_sha256']:raise ValueError('Image content changed since manifest preparation.')
        if r['he_bbox'] != r['ihc_bbox']:
            raise ValueError('Only identity-coordinate pairs are allowed.')
        if r['split'] not in SPLITS:
            raise ValueError('Unknown split: ' + r['split'])
    check_leakage(rows)
    return rows


def check_leakage(rows):
    """A group, image content, or pair may never occur in different splits."""
    seen = {}; content_groups={}; windows=set()
    for r in rows:
        if r['window_id'] in windows:raise ValueError('Duplicate window identity: '+r['window_id'])
        windows.add(r['window_id'])
        for key in ('he_sha256','ihc_sha256'):
            if key not in r:continue
            token=(key,r[key])
            if token in content_groups and content_groups[token]!=r['group_id']:
                raise ValueError('Identical image content has multiple source-group IDs.')
            content_groups[token]=r['group_id']
        for token in [('group', r['group_id']), ('pair', r['pair_id']),
                      ('he_content', r.get('he_sha256', r['he'])),
                      ('ihc_content', r.get('ihc_sha256', r['ihc']))]:
            if token in seen and seen[token] != r['split']:
                raise ValueError('Cross-split leakage detected: ' + str(token))
            seen[token] = r['split']


def automatic_group_split(pairs, seed=0):
    """Keep official test groups untouched; divide the remaining independent groups.

    At least five non-test groups are needed, one per development role. The
    small-group allocation is operational, never evidence of adequate sample size.
    """
    groups = {}
    for p in pairs:
        groups.setdefault(p['group_id'], set()).add(p['split'])
    if any('test' in roles and len(roles) > 1 for roles in groups.values()):
        raise ValueError('An official-test group also occurs outside test.')
    candidates = sorted(g for g, roles in groups.items() if roles != {'test'})
    if len(candidates) < 5:
        raise ValueError('Need at least five non-test source groups, or use the explicit demo command.')
    rng = np.random.default_rng(seed); rng.shuffle(candidates)
    roles = ['train', 'val', 'risk_train', 'calibrate', 'certify']
    allocation = dict(zip(candidates[:5], roles))
    for g in candidates[5:]:
        allocation[g] = rng.choice(roles, p=[.55, .15, .10, .10, .10]).item()
    for p in pairs:
        if p['split'] != 'test':
            p['split'] = allocation[p['group_id']]
    return pairs


def prepare(pairs, output, window=128, min_tissue=.50, min_h_std=.08):
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    rows = [];pair_ids=set()
    for p in pairs:
        if not p.get('group_id') or not p.get('pair_id'):
            raise ValueError('Pair and source-group IDs are required; no guessed patient IDs.')
        if p['pair_id'] in pair_ids:raise ValueError('Duplicate pair ID in input list.')
        pair_ids.add(p['pair_id'])
        he_path, ihc_path = Path(p['he']).resolve(), Path(p['ihc']).resolve()
        he = np.asarray(Image.open(he_path).convert('RGB'))
        ihc = np.asarray(Image.open(ihc_path).convert('RGB'))
        if he.shape != ihc.shape:
            raise ValueError('Paired image shapes differ; resizing is not a correspondence rule.')
        height, width = he.shape[:2]
        if window <= 0 or window % 4 or width % window or height % window:
            raise ValueError('Image dimensions must be divisible by the window; no silent edge crop.')
        # Weak/negative staining is not failed tissue. Use a broad near-white
        # observability test for IHC, independently of DAB positivity.
        hm, im = foreground(he), foreground(ihc,threshold=.02)
        policy=quality_policy(window,min_tissue,min_h_std)
        H, D = components(he, he=True)[:, :, 0], components(ihc)[:, :, 1]
        he_hash, ihc_hash = sha256(he_path), sha256(ihc_path)
        for y in range(0, height, window):
            for x in range(0, width, window):
                sl = np.s_[y:y+window, x:x+window]
                mask = hm[sl]; hc, ic, hs = float(mask.mean()), float(im[sl].mean()), float(H[sl].std())
                reason = 'ok'
                if hc < min_tissue: reason = 'he_foreground'
                elif hs < min_h_std: reason = 'he_structure'
                elif ic < min_tissue: reason = 'reference_unobserved'
                # Shared denominator: only the fixed H&E mask, including at evaluation.
                q = float((D[sl][mask] > .20).mean()) if mask.any() else None
                rows.append(dict(pair_id=p['pair_id'], group_id=p['group_id'],
                    group_kind=p.get('group_kind', 'unknown'), split=p['split'],
                    window_id=f"{p['pair_id']}:R{y//window+1}C{x//window+1}",
                    he=str(he_path), ihc=str(ihc_path), he_sha256=he_hash, ihc_sha256=ihc_hash,
                    he_bbox=[x,y,window,window], ihc_bbox=[x,y,window,window],
                    mapping='identity', he_tissue=hc, ihc_tissue=ic, h_std=hs,
                    qc_policy=policy,protocol_hash=policy_hash(window,policy),
                    valid=reason=='ok', reason=reason, q=q,
                    features=descriptor(he[sl]).tolist()))
    check_leakage(rows)
    output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    return rows


def load_pair_csv(path):
    path = Path(path).resolve()
    with path.open(newline='') as f: rows = list(csv.DictReader(f))
    for r in rows:
        for key in ('he', 'ihc'):
            r[key] = str((path.parent / r[key]).resolve())
    return rows
