"""Arrange existing photos and summarize saved edits; no generation/model inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import datetime as dt
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image,ImageDraw
from skimage.metrics import structural_similarity

ROOT=Path('/n/fs/goose');BASE=ROOT/'experiments/mid0_sae_selectivity';SOURCE=BASE/'additive_20261114_174738_954258'
MULT=ROOT/'layers/runs/mid0_sae_face_001/interventions/stability_top10_4243081'
HASH='cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6'
GROUPS=['black_man','black_woman','white_man','white_woman','east_asian_man','east_asian_woman','middle_eastern_man','middle_eastern_woman','indian_man','indian_woman','hispanic_man','hispanic_woman']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def latest(folder):
    out={}
    for line in (folder/'logs/events.jsonl').read_text().splitlines():
        try:r=json.loads(line);out[r['image_id']]=r
        except ValueError:pass
    return out


@lru_cache(maxsize=60)
def baseline(path):
    with Image.open(path) as im:return np.asarray(im.convert('RGB')).copy()


def evaluate(task):
    r,path,base=task
    with Image.open(path) as image:
        assert image.size==(512,512)
        a=np.asarray(image.convert('RGB')).copy()
    b=baseline(base)
    r=dict(r,image_path=str(path),baseline_image_path=str(base),pixel_mae=float(np.abs(a.astype(np.float32)-b.astype(np.float32)).mean()),
        ssim=float(structural_similarity(b,a,channel_axis=-1,data_range=255)),pixels_identical=bool(np.array_equal(a,b)))
    assert sha(path)==r['image_sha256']
    return r


def main():
    run=BASE/('ten_feature_review_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    for d in ('gallery','results','config','code'):(run/d).mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SOURCE/'config/experiment_config.json').read_text());features=cfg['features'];parent=latest(SOURCE)
    assert cfg['sae_sha256']==HASH
    bases={r['prompt_id']:r for r in parent.values() if r['intervention_type']=='baseline' and r['generation_status']=='complete'}
    rows=[];lookup={};manifest=[]
    for f in features:
        folder=SOURCE/'workers'/f'feature_{f}';events=latest(folder)
        manifest.append(dict(path=str(folder/'logs/events.jsonl'),sha256=sha(folder/'logs/events.jsonl')))
        selected=[r for r in events.values() if r['feature_id']==f and r.get('strength_label')!='zero_control']
        assert len(selected)==420 and all(r['generation_status']=='complete' for r in selected)
        for r in selected:
            lookup[(f,r['group'],r['seed'],r['strength_label'])]=(r,folder/r['image_path'])
            rows.append((r,folder/r['image_path'],SOURCE/bases[r['prompt_id']]['image_path']))
    assert len(rows)==4200
    (run/'config/config.json').write_text(json.dumps(dict(source=str(SOURCE),features=features,groups=GROUPS,seeds=[30,31,32,33,34],checkpoint=cfg['sae_checkpoint'],sha256=HASH,
        gallery_seed=30,gallery_rule='All ten features and all 12 prompt groups at seed30, fixed before viewing outcomes',new_generation=False),indent=2))
    shutil.copy2(__file__,run/'code/old_gallery.py')
    table(run/'config/input_hashes.csv',manifest)
    captions=['Unchanged','Multiply x7','Add +10','Add +20','Add +50'];gallery_rows=[]
    for f in features:
        settings=json.loads((MULT/f'feature_{f}/settings.json').read_text())
        assert settings['sae_weights_sha256']==HASH and settings['block']=='unet.mid_block.attentions.0'
        sheet=Image.new('RGB',(5*256,12*302+65),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,6),f'Feature {f}: SAVED PHOTOS ONLY. Seed 30. Same prompts, checkpoint and mid.0.',fill='black')
        for j,title in enumerate(captions):draw.text((j*256+8,32),title,fill='black')
        for i,group in enumerate(GROUPS):
            pid=f'{group}_seed030';b=SOURCE/bases[pid]['image_path']
            paths=[b,MULT/f'feature_{f}'/'images'/group/'seed030_alpha_p7.png']
            paths +=[lookup[(f,group,30,f'fixed_{v}')][1] for v in (10,20,50)]
            for j,path in enumerate(paths):
                if j==1:
                    control=path.with_name('seed030_alpha_p1.png')
                    assert np.array_equal(baseline(str(b)),baseline(str(control)))
                with Image.open(path) as im:
                    tile=im.copy();tile.thumbnail((256,256));sheet.paste(tile,(j*256,65+i*302+25))
                gallery_rows.append(dict(feature_id=f,group=group,seed=30,column=captions[j],source_path=str(path),source_sha256=sha(path),gallery=f'gallery/feature_{f}.png'))
            draw.text((8,65+i*302),group,fill='black')
        sheet.save(run/'gallery'/f'feature_{f}.png')
        print('Gallery ready',f,flush=True)
    # Compact first look, showing every selected feature without outcome selection.
    for group in ('east_asian_man','black_man'):
        sheet=Image.new('RGB',(3*256,10*300+45),'white');draw=ImageDraw.Draw(sheet)
        for j,title in enumerate(('Unchanged','Multiply x7','Add +50')):draw.text((j*256+5,8),title,fill='black')
        for i,f in enumerate(features):
            b=SOURCE/bases[f'{group}_seed030']['image_path']
            paths=[b,MULT/f'feature_{f}'/'images'/group/'seed030_alpha_p7.png',lookup[(f,group,30,'fixed_50')][1]]
            draw.text((5,45+i*300),f'{group}; feature {f}',fill='black')
            for j,path in enumerate(paths):
                with Image.open(path) as im:tile=im.copy();tile.thumbnail((256,256));sheet.paste(tile,(256*j,45+i*300+24))
        sheet.save(run/'gallery'/f'overview_{group}.png')
    table(run/'results/gallery_sources.csv',gallery_rows)
    intro='# Gallery of existing intervention photos\n\nNo new photos were generated. These sheets arrange saved originals from the previous experiments. Original 512×512 PNGs are untouched; gallery thumbnails are256px derivatives. Seed30 was fixed in advance, and every selected feature and all12 prompt groups are included.\n\n'
    intro+='Columns: unchanged, multiplication ×7, additive +10, +20, +50. Multiplication and addition use different parameterizations; these visual columns are not magnitude-matched. Matched-strength conditions are summarized separately.\n\n'
    intro+='## Quick overview\n\n![East-Asian-man overview](gallery/overview_east_asian_man.png)\n\n![Black-man overview](gallery/overview_black_man.png)\n\n## All prompt groups, by feature\n\n'
    for f in features:intro+=f'### Feature {f}\n\n![Feature {f}, all groups](gallery/feature_{f}.png)\n\n'
    intro+='## Interpretation\n\nThese examples do not establish that additive edits do nothing. Nonzero pixel differences are not automatically semantic changes, and weak or inconsistent semantic changes would not establish that race itself is the cause. Prompt conditioning, other coordinates, spatial structure and distributed model context remain possible explanations. Demographic labels are prompt categories, not biological causes or ground-truth identities.\n'
    (run/'README.md').write_text(intro)
    print('GALLERY_DIRECTORY',run,flush=True)
    # Summarize all existing edits, not only the gallery sample.
    with ProcessPoolExecutor(max_workers=4) as pool:
        measured=[]
        for i,r in enumerate(pool.map(evaluate,rows,chunksize=12)):
            measured.append(r)
            if (i+1)%600==0:print('Existing images evaluated',i+1,flush=True)
    table(run/'results/per_image_pixel_results.csv',measured)
    summaries=[]
    for f in features:
        for label in ('small','moderate','matched_small','matched_moderate','fixed_10','fixed_20','fixed_50'):
            for activity in ('all','active','inactive'):
                s=[r for r in measured if r['feature_id']==f and r['strength_label']==label and (activity=='all' or (not r['original_activation_zero'])==(activity=='active'))]
                if not s:continue
                values=[]
                for seed in range(30,35):
                    z=[r['pixel_mae'] for r in s if r['seed']==seed]
                    if z:values.append(float(np.mean(z)))
                import itertools
                v=np.array(values);ix=np.array(list(itertools.product(range(len(v)),repeat=len(v))));boot=v[ix].mean(1)
                summaries.append(dict(feature=f,strength=label,activity=activity,n_images=len(s),n_seed_clusters=len(values),mean_pixel_mae=float(v.mean()),
                    ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),mean_ssim=float(np.mean([r['ssim'] for r in s])),
                    n_exact_noops=sum(r['pixels_identical'] for r in s),mean_applied_l2=float(np.mean([r['decoded_perturbation_norm'] for r in s]))))
    table(run/'results/feature_strength_summary.csv',summaries)
    (run/'results/complete.json').write_text(json.dumps(dict(edited_images=4200,new_generations=0,features=10,seeds=5)))
    print('ALL EXISTING PIXEL METRICS COMPLETE',run,flush=True)

if __name__=='__main__':main()
