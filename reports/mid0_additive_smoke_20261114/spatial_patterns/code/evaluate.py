"""Per-image SSIM, frozen-region change, naturalness, and existing FairFace audit."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
import torchvision.transforms.functional as TF
BASE=Path('/n/fs/goose/experiments/mid0_sae_selectivity');sys.path.insert(0,str(BASE))
from run import ROOT,digest,atomic_json,csv_write
from collect import collect

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();rows=collect(run)
    spec=importlib.util.spec_from_file_location('audit',ROOT/'FairFace/scripts/audit_fairface_independent.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    from facenet_pytorch import MTCNN
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    weights=ROOT/'ReNO/fairface_weights/res34_fair_align_multi_7_20190809.pt';model=audit.load_fairface(str(weights),device)
    detector=MTCNN(image_size=224,margin=int(224*.25),post_process=False,device='cpu')
    out=run/'evaluations';cache=out/'demographic_cache.jsonl';cached={}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:e=json.loads(line);cached[e['image_id']]=e
            except ValueError:pass
    atomic_json(out/'demographic_evaluator.json',dict(architecture='Existing audit FairFace ResNet34 18 outputs',weights=str(weights),sha256=digest(weights),
        labels=audit.RACE_LABELS,face_selection='largest MTCNN bounding box; all face predictions retained',preprocessing='Existing audit: clamped face crop, PIL resize224, ImageNet normalization',
        versions={p:metadata.version(p) for p in ('torch','torchvision','facenet-pytorch','scikit-image','numpy','Pillow')},
        semantics='Predictions of perceived appearance; no ground-truth race labels',ssim='scikit-image structural_similarity, channel_axis=-1, data_range255; structural similarity proxy, not semantic validity'))
    baselines={r['prompt_id']:r for r in rows if r['method']=='baseline' and r['generation_status']=='complete'}
    demographics=[];metrics=[]
    for r in rows:
        if r['generation_status']!='complete':
            demographics.append(dict(image_id=r['image_id'],status='generation_'+r['generation_status']));continue
        with Image.open(r['image_path']) as im:image=im.convert('RGB');rgb=np.asarray(image).copy()
        old=cached.get(r['image_id'])
        if old and old['image_sha256']==r['image_sha256']:e=old
        else:
            e=dict(image_id=r['image_id'],image_sha256=r['image_sha256'],status='complete',faces=[])
            try:
                boxes,confidences=detector.detect(image);e['mtcnn_detected']=boxes is not None and len(boxes)>0
                if e['mtcnn_detected']:
                    for box,confidence in zip(boxes,confidences):
                        x1,y1,x2,y2=box;face=image.crop((max(0,int(x1)),max(0,int(y1)),min(512,int(x2)),min(512,int(y2))))
                        x=TF.to_tensor(face.resize((224,224))).unsqueeze(0).to(device);x=TF.normalize(x,[.485,.456,.406],[.229,.224,.225])
                        with torch.no_grad():logits=model(x)[0].cpu()
                        e['faces'].append(dict(box=box.tolist(),detection_confidence=float(confidence),logits=logits.tolist(),race_probabilities=logits[:7].softmax(0).tolist(),gender_probabilities=logits[7:9].softmax(0).tolist()))
                    face=max(e['faces'],key=lambda f:(f['box'][2]-f['box'][0])*(f['box'][3]-f['box'][1]))
                    e.update(race_probabilities=face['race_probabilities'],gender_probabilities=face['gender_probabilities'],predicted_group=audit.RACE_LABELS[int(np.argmax(face['race_probabilities']))])
            except Exception:e.update(status='failed',error=traceback.format_exc())
            with cache.open('a') as f:f.write(json.dumps(e)+'\n');f.flush()
        demographics.append(e)
        with Image.open(baselines[r['prompt_id']]['image_path']) as b:reference=np.asarray(b.convert('RGB')).copy()
        with np.load(run/'masks'/(r['prompt_id']+'.npz')) as a:mask=a['full_mask'];coarse=a['pattern']
        diff=np.abs(rgb.astype(np.float64)-reference.astype(np.float64)).mean(-1)
        m=dict(r,pixel_mae=float(diff.mean()),ssim=float(structural_similarity(reference,rgb,channel_axis=-1,data_range=255)),
            lowerface_mae=float((diff*mask).sum()/mask.sum()),outside_lowerface_mae=float((diff*(1-mask)).sum()/(1-mask).sum()),
            lowerface_change_fraction=float((diff*mask).sum()/max(diff.sum(),1e-12)),
            outside_change_fraction=float((diff*(1-mask)).sum()/max(diff.sum(),1e-12)),semantic_facial_hair_status='pending_blinded_review')
        if r['method'] not in ('baseline',):
            with np.load(r['activation_path']) as z:
                p=z['base_pattern'][0,...,0];change=z['latent_change'][0,...,0];energy=change.astype(float)**2
                modified=z['modified_latents'];before=z['original_latents'];after=z['modified_feature']
                m.update(pattern_nonzero_cells=int(np.count_nonzero(p)),pattern_energy_in_lowerface=float((p*p*coarse).sum()/max(float((p*p).sum()),1e-12)),
                    change_effective_positions=float(energy.sum()**2/max(float((energy**2).sum()),1e-12)),
                    max_latent_change=float(np.abs(change).max()),original_support_mean=float(np.count_nonzero(before,axis=-1).mean()),
                    modified_support_mean=float(np.count_nonzero(modified,axis=-1).mean()),fraction_positions_exceeding_topk=float(np.mean(np.count_nonzero(modified,axis=-1)>10)),
                    modified_feature_negative_fraction=float(np.mean(after<0)),applied_tensor_l2=float(np.linalg.norm(z['applied_block_delta'])),
                    latent_change_l2=float(np.linalg.norm(change)),post_reencoded_discrepancy=float(np.linalg.norm(z['post_reencoded_feature']-after)))
                assert abs(m['applied_tensor_l2']-r['actual_applied_norm'])<.01
                if r['method']!='no_op':assert r['match_passed']
        metrics.append(m)
    byid={e['image_id']:e for e in demographics}
    for m in metrics:
        e=byid[m['image_id']];b=byid.get(baselines[m['prompt_id']]['image_id'],{})
        m['demographic_tv']=float(np.abs(np.asarray(e['race_probabilities'])-b['race_probabilities']).sum()/2) if 'race_probabilities' in e and 'race_probabilities' in b else None
    csv_write(out/'demographic_predictions.csv',[dict(e,faces=json.dumps(e.get('faces')),race_probabilities=json.dumps(e.get('race_probabilities')),gender_probabilities=json.dumps(e.get('gender_probabilities'))) for e in demographics])
    atomic_json(out/'demographic_predictions.json',demographics)
    csv_write(out/'image_metrics.csv',metrics);atomic_json(out/'image_metrics.json',metrics)
    csv_write(out/'segmentation_results.csv',[dict(status='not_used_for_beard_scoring',reason='Existing SAM probes are class-agnostic and lack validated beard masks; use blinded human ratings')])
    print('Metrics',len(metrics),'demographic rows',len(demographics))
if __name__=='__main__':main()
