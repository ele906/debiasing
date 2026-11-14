"""Use the repository's landmark-derived ROI; no beard classification is performed."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import ROOT,digest,atomic_json,csv_write

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    path=ROOT/'face-landmarker/render_landmarked_lower_face_change_report.py'
    spec=importlib.util.spec_from_file_location('existing_roi',path);roi=importlib.util.module_from_spec(spec);sys.modules[spec.name]=roi;spec.loader.exec_module(roi)
    model=ROOT/'face-landmarker/models/face_landmarker.task';mp,detector=roi.build_detector(model,ROOT/'face-landmarker/.deps')
    baseline=json.loads((run/'config/baselines.json').read_text());rows=[]
    with detector:
        for pid,b in baseline.items():
            dst=run/'masks'/(pid+'.npz')
            if dst.exists():raise FileExistsError('Masks already frozen; do not overwrite '+str(dst))
            result=detector.detect(mp.Image.create_from_file(str(run/b['image_path'])))
            if not result.face_landmarks:raise ValueError('No landmarks for recipient '+pid+'; no silent mask fallback')
            full,polygon=roi.beard_roi_from_landmarks(result.face_landmarks[0],512,512)
            coarse=np.asarray(Image.fromarray(full.astype(np.float32)).resize((16,16),resample=Image.Resampling.BOX),dtype=np.float32)
            assert np.any(coarse) and full.shape==(512,512)
            landmarks=np.array([[l.x,l.y,l.z] for l in result.face_landmarks[0]],np.float32)
            np.savez_compressed(dst,pattern=coarse,full_mask=full,landmarks=landmarks,polygon=np.asarray(polygon),layer=np.asarray('unet.mid_block.attentions.0'))
            Image.fromarray(np.uint8(full*255)).save(run/'masks'/(pid+'_mask.png'))
            rows.append(dict(prompt_id=pid,mask_path=str(dst.relative_to(run)),mask_sha256=digest(dst),full_area=float(full.sum()),coarse_nonzero_cells=int(np.count_nonzero(coarse)),
                             baseline_detected=True,mediapipe_version=mp.__version__))
    csv_write(run/'config/masks.csv',rows)
    atomic_json(run/'config/mask_evaluator.json',dict(model=str(model),sha256=digest(model),mediapipe=mp.__version__,numpy=np.__version__,
        detector='FaceLandmarker IMAGE, num_faces=1, default thresholds .5',roi_function='beard_roi_from_landmarks',resize='float32 Pillow BOX area average; fractional weights preserved',
        meaning='Geometric lower-face ROI, not semantic beard segmentation'))
    print('Frozen masks',len(rows))
if __name__=='__main__':main()
