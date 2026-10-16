"""Restore ignored test attachments/weights from the previously delivered ZIP.

Uses only the standard library. Never imports code from the ZIP into this Git
checkout. The pinned SHA256 prevents accidentally using a different snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def restore(archive,replace=False):
    archive=Path(archive).resolve()
    spec=json.loads((ROOT/'repro_assets.json').read_text())
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=spec['sha256']:
        raise ValueError('Archive SHA256 differs from repro_assets.json; no files were restored.')
    entries=[]
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if entry.is_dir():continue
            parts=PurePosixPath(entry.filename).parts
            if '\\' in entry.filename or '..' in parts or entry.filename.startswith('/'):
                raise ValueError('Unsafe archive path.')
            if len(parts)<3 or parts[0]!='contrastive-unpaired-translation-master':continue
            relative=PurePosixPath(*parts[1:])
            if relative.parts[0] not in {'examples','artifacts'}:continue
            if str(relative)=='examples/pairs_template.csv':continue
            if stat.S_ISLNK(entry.external_attr>>16):raise ValueError('Symbolic links are not imported.')
            target=(ROOT/str(relative)).resolve()
            if not target.is_relative_to(ROOT):raise ValueError('Destination escapes the checkout.')
            entries.append((entry,target))
        if sum(x.file_size for x,_ in entries)>1024**3:raise ValueError('Unexpected asset size.')
        # Validate collisions before making any changes.
        for entry,target in entries:
            if target.exists() and not replace:
                if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(z.read(entry)).digest():
                    raise ValueError(f'Existing asset differs: {target.relative_to(ROOT)}; use --replace intentionally.')
        for entry,target in entries:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(entry))
    return dict(restored_files=len(entries),archive_sha256=digest.hexdigest(),
                destination=str(ROOT),note='Assets remain excluded from Git.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True)
    p.add_argument('--replace',action='store_true',help='Explicitly replace differing local ignored assets.')
    a=p.parse_args();print(json.dumps(restore(a.archive,a.replace),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
