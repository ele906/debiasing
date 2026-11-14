"""Summarize fixed feature sweeps using unreviewed detector presence proxies."""
import argparse
import csv
from pathlib import Path
import re
import numpy as np


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path(__file__).resolve().parent/'run_001')
    args=p.parse_args()
    metrics=read(args.run/'metrics.csv')
    predictions=read(args.run/'detector/concept_predictions.csv')
    present={r['sample_id']:any(re.search(r'\bnecklace\b',s.lower()) for s in r['detected_concepts'].split(';')) for r in predictions}
    groups={}
    for r in metrics:groups.setdefault((r['role'],r['feature'],r['strength']),[]).append(r)
    rows=[]
    for (role,feature,strength),samples in groups.items():
        before=np.array([present[r['sample_id']+'_baseline'] for r in samples])
        after=np.array([present[r['edited_sample_id']] for r in samples])
        means=lambda key: '' if not any(r[key]!='' for r in samples) else float(np.mean([float(r[key]) for r in samples if r[key]!='']))
        rows.append(dict(role=role,feature=feature,strength=strength,images=len(samples),active_baseline_images=sum(float(r['baseline_feature_mean'])>0 for r in samples),baseline_detector_positive=int(before.sum()),edited_detector_positive=int(after.sum()),detector_presence_lost=int((before&~after).sum()),detector_presence_gained=int((~before&after).sum()),mean_pixel_mae=means('pixel_mae'),mean_necklace_box_mae=means('necklace_box_mae'),mean_face_box_mae=means('face_box_mae')))
    with (args.run/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    report=['# Tuning sweep summary','','Detector presence is an unreviewed proxy. Lost labels do not establish concept removal or reduced racism. Requested present/absent conditions can disagree with what was actually generated.','', '| Role | Feature | Alpha | Images | Baseline labels | Edited labels | Labels lost | Labels gained | Mean pixel change |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:report.append(f'| {r["role"]} | {r["feature"]} | {r["strength"]} | {r["images"]} | {r["baseline_detector_positive"]} | {r["edited_detector_positive"]} | {r["detector_presence_lost"]} | {r["detector_presence_gained"]} | {r["mean_pixel_mae"]:.3f} |')
    report += ['', 'Pixel MAE is on RGB values from 0 to 255. Region measurements use the baseline detector boxes. The control is matched approximately on activity; it is not known to be semantically neutral. Inspect target/control grids and review.csv for necklace change, face/clothing damage, artifacts, and prompt fidelity. No best alpha has been selected.','', '[All comparison grids](REPORT.md)', '[Full measurements](metrics.csv)', '[Blank review sheet](review.csv)']
    (args.run/'SUMMARY.md').write_text('\n'.join(report).rstrip()+'\n')
    print('\n'.join(report),flush=True)


if __name__=='__main__':main()
