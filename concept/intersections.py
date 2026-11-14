"""Validate detector/SAE candidate intersections on spatial validation maps."""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def coverage(boxes,image_size,shape):
    """Exact image-pixel union coverage, averaged over SAE spatial cells."""
    width,height=image_size
    mask=np.zeros((height,width),dtype=np.uint8)
    for box in boxes:
        x1,y1,x2,y2=box
        if not np.isfinite(box).all() or x2<x1 or y2<y1:continue
        x1,y1=min(width,max(0,int(np.floor(x1)))),min(height,max(0,int(np.floor(y1))))
        x2,y2=min(width,max(0,int(np.ceil(x2)))),min(height,max(0,int(np.ceil(y2))))
        mask[y1:y2,x1:x2]=1
    gh,gw=shape
    if height%gh or width%gw:raise ValueError('Image/grid dimensions must divide exactly')
    return mask.reshape(gh,height//gh,gw,width//gw).mean(axis=(1,3))


def localization(values,mask):
    total=float(values.sum());area=float(mask.mean())
    if total<=0 or area<=0:return dict(box_area_fraction=area,activation_mass_in_box='',spatial_enrichment='',shift_reference_fraction='')
    observed=float((values*mask).sum()/total)
    shifted=[float((np.roll(values,(dy,dx),axis=(0,1))*mask).sum()/total) for dy in range(values.shape[0]) for dx in range(values.shape[1])]
    return dict(box_area_fraction=area,activation_mass_in_box=observed,spatial_enrichment=observed/area,shift_reference_fraction=float(np.mean(np.array(shifted)>=observed-1e-12)))


def boxes_for(record,pattern):
    boxes=[]
    for source in ['grounding','objects']:
        result=record[source]
        for label,box in zip(result['labels'],result['bboxes']):
            if re.search(pattern,str(label).lower()) and len(box)==4 and np.isfinite(box).all() and box[2]>=box[0] and box[3]>=box[1]:boxes.append(box)
    return boxes


def extract(args):
    settings=json.loads((ROOT/'backprop/generation_settings.json').read_text())
    comparison=json.loads((HERE/'sae_comparison_full/settings.json').read_text())
    summary=read(HERE/'sae_comparison_full/summary.csv')
    source=read(HERE/'sae_comparison_full/candidates.csv') if args.all_top_candidates else summary
    selected=[r for r in source if r['feature']!='' and (not args.concepts or r['concept'] in args.concepts)]
    candidates=[dict(concept=r['concept'],feature=int(r['feature']),pattern=comparison['patterns'][r['concept']]) for r in selected]
    rows=[r for r in read(ROOT/'backprop/manifest.csv') if r['split']=='validation']
    if len(rows)!=240:raise ValueError('Expected 240 validation images')
    checkpoint=Path(settings['sae_checkpoint'])
    if hashlib.sha256((checkpoint/'state_dict.pth').read_bytes()).hexdigest()!=settings['sae_weights_sha256']:raise ValueError('Checkpoint mismatch')
    provenance=dict(min_group_count=comparison.get('min_group_count',3),candidate_source='all_top_candidates' if args.all_top_candidates else 'top_one',candidates=candidates,generation=settings,predictions_sha256=hashlib.sha256((HERE/'concept_predictions.csv').read_bytes()).hexdigest(),comparison_sha256=hashlib.sha256((HERE/'sae_comparison_full/summary.csv').read_bytes()).hexdigest(),split='validation',samples=len(rows),box_reference='Fractional spatial cell coverage of union of matching detector boxes',localization_reference='All circular translations of each SAE map; exploratory reference, not a calibrated significance test')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'settings.json'
    if path.exists() and json.loads(path.read_text())!=provenance:raise ValueError('Settings changed; choose a fresh output')
    path.write_text(json.dumps(provenance,indent=2)+'\n')
    (args.output/'maps').mkdir(exist_ok=True)
    sys.path.insert(0,str(ROOT/'sdxl-unbox'))
    import torch
    from diffusers import AutoencoderKL,EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder
    torch.set_num_threads(4)
    load=dict(torch_dtype=torch.float16,cache_dir=str(ROOT/'ReNO/hf_cache'),local_files_only=True)
    print('Loading cached SDXL Turbo and fixed SAE',flush=True)
    vae=AutoencoderKL.from_pretrained(settings['vae'],**load)
    pipe=HookedStableDiffusionXLPipeline.from_pretrained(settings['model'],vae=vae,variant='fp16',**load)
    pipe.pipe.scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config,timestep_spacing='trailing')
    pipe.pipe.to('cuda');pipe.pipe.set_progress_bar_config(disable=True)
    sae=SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval();sae.requires_grad_(False)
    metrics=[]
    ids=[c['feature'] for c in candidates]
    block=settings['block']
    for i,row in enumerate(rows):
        sample=row['sample_id'];save=args.output/'maps'/(sample+'.npz')
        if save.exists():
            with np.load(save,allow_pickle=False) as f:
                maps=f['maps'];error=float(f['replay_mae'])
                if not np.array_equal(f['features'],ids):raise ValueError('Saved feature IDs differ')
        else:
            seed=int(row['seed'])
            noise=torch.randn((1,4,64,64),device='cuda',dtype=torch.float16,generator=torch.Generator('cuda').manual_seed(seed))
            with torch.inference_mode():
                result,cache=pipe.run_with_cache(prompt=row['prompt'],latents=noise,generator=torch.Generator('cuda').manual_seed(seed),num_inference_steps=1,guidance_scale=0.,height=512,width=512,positions_to_cache=[block],save_input=True)
                residual=(cache['output'][block]-cache['input'][block])[:,-1].permute(0,2,3,1).float()
                maps=sae.encode(residual)[0,...,ids].permute(2,0,1).cpu().numpy()
            with Image.open(ROOT/'backprop'/row['image']) as src:original=np.asarray(src.convert('RGB'))
            error=float(np.abs(original.astype(float)-np.asarray(result.images[0])).mean())
            if error!=0:raise RuntimeError(f'Replay mismatch: {sample}, MAE {error}')
            np.savez_compressed(save,maps=maps,features=np.array(ids),replay_mae=np.array(error))
            del cache,residual
        record=json.loads((HERE/'full_run/records'/(sample+'.json')).read_text())
        if hashlib.sha256((ROOT/'backprop'/row['image']).read_bytes()).hexdigest()!=record['image_sha256']:raise ValueError('Detector image changed')
        for c,values in zip(candidates,maps):
            boxes=boxes_for(record,c['pattern'])
            phrases=[s for source in ['grounding','objects'] for s in record[source]['labels']]
            present=any(re.search(c['pattern'],str(s).lower()) for s in phrases)
            mask=coverage(boxes,(512,512),values.shape)
            metrics.append(dict(sample_id=sample,race_prompt=row['race_prompt'],gender_prompt=row['gender_prompt'],concept=c['concept'],feature=c['feature'],detector_present=int(present),valid_matching_boxes=len(boxes),feature_mean=float(values.mean()),feature_max=float(values.max()),feature_active_fraction=float((values>0).mean()),replay_mae=error,**localization(values,mask)))
        if i%20==0 or i==len(rows)-1:print(f'{i+1}/{len(rows)} validation images spatially checked',flush=True)
    write(args.output/'spatial_metrics.csv',metrics)
    report(args)


def report(args):
    metrics=read(args.output/'spatial_metrics.csv')
    settings=json.loads((args.output/'settings.json').read_text())
    source='candidates.csv' if settings.get('candidate_source')=='all_top_candidates' else 'summary.csv'
    summary=read(HERE/'sae_comparison_full'/source)
    candidates={(c['concept'],str(c['feature'])):c for c in settings['candidates']}
    summary=[r for r in summary if (r['concept'],r['feature']) in candidates]
    manifest={r['sample_id']:r for r in read(ROOT/'backprop/manifest.csv')}
    rows=[]
    review_path=HERE/'intersections/visual_review.csv'
    reviewed_misses={r['concept'] for r in read(review_path) if r['detector_label']=='0'} if review_path.exists() else set()
    for s in summary:
        concept=s['concept']
        positive=[r for r in metrics if r['concept']==concept and r['feature']==s['feature'] and r['detector_present']=='1']
        localized=[r for r in positive if r['spatial_enrichment']!='']
        enrich=np.array([float(r['spatial_enrichment']) for r in localized])
        fractions=np.array([float(r['shift_reference_fraction']) for r in localized])
        val_auc=float(s['validation_matched_auc']) if s['validation_matched_auc'] else None
        median=float(np.median(enrich)) if len(enrich) else None
        group_rows={}
        for r in metrics:
            if r['concept']==concept and r['feature']==s['feature']:
                group_rows.setdefault((r['race_prompt'],r['gender_prompt']),[]).append(r)
        qualified=[group for group in group_rows.values() if min(sum(r['detector_present']=='1' for r in group),sum(r['detector_present']=='0' for r in group))>=settings.get('min_group_count',3)]
        controlled=[r for group in qualified for r in group if r['detector_present']=='1' and r['spatial_enrichment']!='']
        controlled_median=float(np.median([float(r['spatial_enrichment']) for r in controlled])) if controlled else None
        status='not supported by current checks'
        if val_auc is None:status='insufficient matched validation controls'
        elif val_auc>.5 and controlled_median is not None and controlled_median>1:status='provisional activation + controlled spatial intersection'
        screen_pass=int(val_auc is not None and val_auc>.5 and controlled_median is not None and controlled_median>1)
        if concept in reviewed_misses:status='detector-negative controls need human review'
        rows.append(dict(concept=concept,feature=s['feature'],candidate_train_rank=s.get('rank','1'),validation_auc=s['validation_auc'],validation_matched_auc=s['validation_matched_auc'],validation_matched_groups=s['validation_matched_groups'],detector_positive_images=len(positive),positive_images_with_boxes=sum(int(r['valid_matching_boxes'])>0 for r in positive),active_positive_images=sum(float(r['feature_max'])>0 for r in positive),localized_active_images=len(localized),median_spatial_enrichment='' if median is None else median,controlled_active_positive_images=len(controlled),controlled_median_spatial_enrichment='' if controlled_median is None else controlled_median,images_above_area_baseline=int((enrich>1).sum()),median_shift_reference_fraction='' if not len(fractions) else float(np.median(fractions)),numeric_screen_pass=screen_pass,status=status))
    write(args.output/'intersection_summary.csv',rows)
    figures=args.output/'figures';figures.mkdir(exist_ok=True)
    sys.path.insert(0,str(ROOT/'backprop'))
    reports=['# C2C–SAE candidate intersections','','All 240 validation images were replayed with exact pixel identity. No SAE or diffusion training or suppression. Feature IDs were fixed by the earlier train-only ranking. Detector labels and boxes remain unreviewed.','', 'Spatial enrichment = fraction of feature activation inside matching boxes / fraction of image covered by those boxes. Uniform spatial activation has enrichment 1. Zero-activation maps are excluded from localization ratios and counted separately. Box coverage accounts for coarse SAE cells. Circular map translations provide an exploratory spatial reference, not a significance test.','', '| Concept | Feature | Within-group validation AUC | Detector positives | Active positives | Pooled spatial enrichment | Controlled spatial enrichment | Status |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        fmt=lambda v: '' if v=='' else f'{float(v):.3f}'
        reports.append(f'| {r["concept"]} | {r["feature"]} | {fmt(r["validation_matched_auc"])} | {r["detector_positive_images"]} | {r["active_positive_images"]} | {fmt(r["median_spatial_enrichment"])} | {fmt(r["controlled_median_spatial_enrichment"])} | {r["status"]} |')
    reports += ['', 'The status is a descriptive screen: within-group validation AUC above 0.5 and median spatial enrichment above 1 inside the same qualifying validation groups (at least three detector positives and three negatives). Pooled enrichment from other groups cannot validate that intersection. It is not semantic ground truth or causal evidence. A concept with one qualifying group has limited controls. Many features/concepts are inspected, and validation has been used iteratively; confirmatory conclusions need new independent examples.', '', '## Matched visual examples', '', 'Each figure uses one race/gender prompt group containing both detector-positive and detector-negative validation images. Choose the group with the most balanced classes, then the first two examples per class in manifest order. This selection does not use SAE activation or overlap. Blue boxes are detector regions; the right panel combines boxes and the SAE map. Maps share one scale across the four examples for a concept. Negatives can be detector misses.']
    for (concept,feature),c in candidates.items():
        eligible=[r for r in metrics if r['concept']==concept and r['feature']==feature]
        groups={}
        for r in eligible:groups.setdefault((r['race_prompt'],r['gender_prompt']),[]).append(r)
        controls=[(min(sum(r['detector_present']=='1' for r in group),sum(r['detector_present']=='0' for r in group)),key,group) for key,group in groups.items()]
        controls=[r for r in controls if r[0]>0]
        if not controls:continue
        _,group,examples=sorted(controls,key=lambda item:(-item[0],item[1]))[0]
        selected=[]
        for flag in ['1','0']:selected.extend([r for r in examples if r['detector_present']==flag][:2])
        maps=[]
        for r in selected:
            with np.load(args.output/'maps'/(r['sample_id']+'.npz'),allow_pickle=False) as f:maps.append(f['maps'][list(f['features']).index(int(c['feature']))])
        vmax=max(float(v.max()) for v in maps)
        panels=[]
        for r,values in zip(selected,maps):
            source=manifest[r['sample_id']]
            with Image.open(ROOT/'backprop'/source['image']) as src:image=src.convert('RGB')
            record=json.loads((HERE/'full_run/records'/(r['sample_id']+'.json')).read_text())
            boxes=boxes_for(record,c['pattern'])
            # Construct the fixed-scale map directly on the original image.
            from matplotlib import colormaps
            norm=values/vmax if vmax>0 else np.zeros_like(values)
            rgba=colormaps['jet'](norm);rgba[...,3]=np.where(values>0,.6,0)
            overlay=Image.alpha_composite(image.convert('RGBA'),Image.fromarray((rgba*255).astype(np.uint8)).resize(image.size,Image.Resampling.NEAREST)).convert('RGB')
            left=image.copy();ld=ImageDraw.Draw(left);rd=ImageDraw.Draw(overlay)
            for box in boxes:ld.rectangle(box,outline='blue',width=3);rd.rectangle(box,outline='blue',width=3)
            panel=Image.new('RGB',(768,450),'white');d=ImageDraw.Draw(panel)
            label=f'{r["sample_id"]}: detector={r["detector_present"]}, SAE={c["feature"]}'
            d.text((4,4),label,fill='black')
            d.text((4,20),f'mean={float(r["feature_mean"]):.4g}, enrichment={r["spatial_enrichment"] or "undefined"}',fill='black')
            panel.paste(left.resize((384,384)),(0,48));panel.paste(overlay.resize((384,384)),(384,48))
            panels.append(panel)
        sheet=Image.new('RGB',(768,450*len(panels)),'white')
        for i,panel in enumerate(panels):sheet.paste(panel,(0,450*i))
        sheet.save(figures/(concept+'_feature'+feature+'.jpg'),quality=90)
        reports += ['', f'### {concept}, feature {feature} — {group[0]} {group[1]}', '', f'![{concept} detector and SAE comparison](figures/{concept}_feature{feature}.jpg)']
    reports += ['', '## Visual review findings', '', 'The top-one necklace feature 3985 activated around the mouth in the inspected matched Indian-man examples, rather than the necklace. The two inspected detector-negative afro examples (black_woman_seed042 and black_woman_seed051) visibly contain the hairstyle. The two inspected detector-negative beard examples (east_asian_man_seed002 and east_asian_man_seed012) visibly contain facial hair. These are spot checks, not a complete annotation pass; raw detector labels are preserved. Afro and beard controls cannot support a confirmed semantic distinction until corrected. numeric_screen_pass records the quantitative screen separately from these observed detector-quality issues.', '', '## What remains', '', 'Review/correct boxes and labels, inspect whether features localize consistently across prompt groups, collect independent controls, then test attenuation with random-feature controls. Spatial overlap alone can result from nearby or correlated content and does not establish that a feature exclusively represents the concept.']
    (args.output/'REPORT.md').write_text('\n'.join(reports).rstrip()+'\n')
    print('Intersection summary and matched figures complete',flush=True)
    for r in rows:print(r,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'intersections');p.add_argument('--report-only',action='store_true');p.add_argument('--all-top-candidates',action='store_true');p.add_argument('--concepts',nargs='+');args=p.parse_args()
    report(args) if args.report_only else extract(args)
