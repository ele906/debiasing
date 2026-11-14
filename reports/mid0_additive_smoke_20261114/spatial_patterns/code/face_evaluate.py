"""Existing MediaPipe detector and per-image landmark geometry; not semantic facial validity."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
BASE=Path('/n/fs/goose/experiments/mid0_sae_selectivity');sys.path.insert(0,str(BASE))
from run import ROOT,digest,atomic_json,csv_write
from collect import collect

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();rows=collect(run)
    sys.path.append(str(ROOT/'face-landmarker/.deps'))
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    model=ROOT/'face-landmarker/models/face_landmarker.task'
    options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model)),num_faces=1,
        min_face_detection_confidence=.5,min_face_presence_confidence=.5,min_tracking_confidence=.5)
    out=run/'evaluations';(out/'landmarks').mkdir(parents=True,exist_ok=True);cache=out/'landmark_cache.jsonl';cached={}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:e=json.loads(line);cached[e['image_id']]=e
            except ValueError:pass
    atomic_json(out/'landmark_evaluator.json',dict(model=str(model),model_sha256=digest(model),mediapipe_version=mp.__version__,numpy_version=np.__version__,
        min_detection_confidence=.5,min_presence_confidence=.5,min_tracking_confidence=.5,structural_validity='Manual review required; detection and geometry alone are not facial validity'))
    results=[]
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for r in rows:
            if r['generation_status']!='complete':
                results.append(dict(image_id=r['image_id'],status='generation_'+r['generation_status'],face_detected=None));continue
            old=cached.get(r['image_id'])
            if old and old['image_sha256']==r['image_sha256']:results.append(old);continue
            e=dict(image_id=r['image_id'],image_sha256=r['image_sha256'],status='complete',face_detected=None,structural_validity='unreviewed')
            try:
                with Image.open(r['image_path']) as image:rgb=np.asarray(image.convert('RGB'))
                result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
                e['face_detected']=bool(result.face_landmarks)
                if result.face_landmarks:
                    landmarks=np.asarray([[l.x,l.y,l.z] for l in result.face_landmarks[0]],np.float32)
                    path=out/'landmarks'/(r['image_id']+'.npz');np.savez_compressed(path,landmarks=landmarks)
                    e.update(landmark_path=str(path.relative_to(run)),landmark_sha256=digest(path),landmark_count=len(landmarks),
                        landmark_nonfinite=not bool(np.isfinite(landmarks).all()),xy_outside_fraction=float(np.mean((landmarks[:,:2]<0)|(landmarks[:,:2]>1))))
            except Exception as exc:e.update(status='failed',error=repr(exc))
            with cache.open('a') as f:f.write(json.dumps(e)+'\n');f.flush()
            results.append(e)
    csv_write(out/'face_validity.csv',results)
    print('MediaPipe images',len(results),'detected',sum(e.get('face_detected') is True for e in results))
if __name__=='__main__':main()
