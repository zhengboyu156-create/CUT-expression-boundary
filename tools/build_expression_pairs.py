"""Build image-pair CSV automatically by identity keys; no spatial searching.

Metadata CSV columns: pair_id,group_id,group_kind,split. Source identifiers must
come from dataset metadata, not from pretending every patch is a patient.
"""
import argparse
import csv
from pathlib import Path
import re


def indexed(directory,pattern):
    result={}
    for p in sorted(Path(directory).rglob('*')):
        if p.suffix.lower() not in ('.png','.jpg','.jpeg','.tif','.tiff','.bmp'):continue
        match=re.fullmatch(pattern,p.stem)
        if not match:raise ValueError('Filename does not match identity pattern: '+str(p))
        key=match.group(1)
        if key in result:raise ValueError('Duplicate pair identity: '+key)
        result[key]=str(p.resolve())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--he-dir',required=True);p.add_argument('--ihc-dir',required=True)
    p.add_argument('--metadata',required=True);p.add_argument('--out',required=True)
    p.add_argument('--he-key-regex',default='(.*)');p.add_argument('--ihc-key-regex',default='(.*)')
    a=p.parse_args();he=indexed(a.he_dir,a.he_key_regex);ihc=indexed(a.ihc_dir,a.ihc_key_regex)
    if he.keys()!=ihc.keys():raise ValueError('Unpaired identities; no silent nearest-image substitution.')
    with open(a.metadata,newline='') as f:metadata={r['pair_id']:r for r in csv.DictReader(f)}
    if metadata.keys()!=he.keys():raise ValueError('Metadata and image pair identities must match exactly.')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['pair_id','group_id','group_kind','split','he','ihc']);writer.writeheader()
        for key in sorted(he):writer.writerow({**metadata[key],'he':he[key],'ihc':ihc[key]})
    print('Paired identities:',len(he))


if __name__=='__main__':main()
