"""Prepare four full-dataset extraction shards, then merge verified outputs."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from extract import summarize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FULL=HERE/'full_run'


def prepare():
    if FULL.exists():raise ValueError('Full extraction inputs already exist')
    FULL.mkdir()
    with (ROOT/'backprop/manifest.csv').open() as f:rows=list(csv.DictReader(f))
    for row in rows:
        for field in ['image','features']:row[field]=str((ROOT/'backprop'/row[field]).resolve())
        if not Path(row['image']).is_file():raise ValueError('Missing source image')
    if len(rows)!=1200 or len({r['sample_id'] for r in rows})!=1200:raise ValueError('Expected 1,200 unique images')
    for shard in range(4):
        directory=FULL/f'shard_{shard}';directory.mkdir()
        subset=rows[shard::4]
        with (directory/'input.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(subset)
        # Reuse pilot extraction only where exact image hash matches.
        for row in subset:
            old=HERE/'extracted'/(row['sample_id']+'.json')
            if old.exists():
                record=json.loads(old.read_text())
                if record['image_sha256']!=hashlib.sha256(Path(row['image']).read_bytes()).hexdigest():raise ValueError('Pilot image mismatch')
                shutil.copy2(old,directory/old.name)
                overlay=old.with_name(old.stem+'_grounded.jpg')
                if overlay.exists():shutil.copy2(overlay,directory/overlay.name)
    with (FULL/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('Prepared four shards of 300 images; reused 72 pilot records',flush=True)


def merge():
    with (FULL/'manifest.csv').open() as f:rows=list(csv.DictReader(f))
    records=[]
    for i,row in enumerate(rows):
        source=FULL/f'shard_{i%4}'/(row['sample_id']+'.json')
        record=json.loads(source.read_text())
        if record['sample_id']!=row['sample_id']:raise ValueError('Sample mismatch')
        if record['image_sha256']!=hashlib.sha256(Path(row['image']).read_bytes()).hexdigest():raise ValueError('Image mismatch')
        for task in ['grounding','objects']:
            if len(record[task]['labels'])!=len(record[task]['bboxes']):raise ValueError('Box mismatch')
        records.append(record)
    # Preserve the pilot summaries before publishing full summaries at top level.
    archive=HERE/'pilot/extraction_summary';archive.mkdir(exist_ok=False)
    for name in ['REPORT.md','concept_predictions.csv','concept_frequencies.csv']:
        shutil.copy2(HERE/name,archive/name)
    flat=FULL/'records';flat.mkdir(exist_ok=False)
    for row,record in zip(rows,records):
        (flat/(row['sample_id']+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    summarize(rows,records,HERE)
    (FULL/'COMPLETED.json').write_text(json.dumps(dict(images=len(records),model_revision=(HERE/'model_revision.txt').read_text().strip(),split_counts={s:sum(r['split']==s for r in rows) for s in ['train','validation','test']}),indent=2)+'\n')
    print('Merged and verified 1,200 images; top-level summaries now cover the full dataset',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','merge']);a=p.parse_args()
    prepare() if a.command=='prepare' else merge()
