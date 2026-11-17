"""Pure storage and prompt geometry helpers for a frozen-SAM pilot."""
import csv,hashlib,json,os
from pathlib import Path
import numpy as np
ROOT=Path('/n/fs/goose')
SAE_HASH='cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6'

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def save_json(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)

def csv_write(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 if not keys:keys=['status'];rows=[dict(status='no_rows')]
 tmp=path.with_suffix('.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 os.replace(tmp,path)

def prompt_geometry(landmarks,width,height):
 p=np.asarray(landmarks,dtype=float)[:,:2]*[width,height]
 left,right=sorted((p[234],p[454]),key=lambda q:q[0]);chin=p[152];brow=p[10];lip=p[17]
 fw=max(float(right[0]-left[0]),1.);fh=max(float(chin[1]-brow[1]),1.);center=(left[0]+right[0])/2
 span=max(float(chin[1]-lip[1]),.08*fh)
 box=np.array([left[0]-.05*fw,lip[1]-.08*fh,right[0]+.05*fw,chin[1]+.42*fh])
 box=np.maximum(box,[0,0,0,0]);box=np.minimum(box,[width-1,height-1,width-1,height-1])
 if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Invalid landmark-derived box')
 foreground=np.array([[center-.28*fw,lip[1]+.65*span],[center+.28*fw,lip[1]+.65*span],
     [center-.13*fw,lip[1]+.90*span],[center+.13*fw,lip[1]+.90*span],[center,chin[1]+.03*fh]])
 background=np.array([(p[13]+p[14])/2,[center-.28*fw,lip[1]-.17*fh],[center+.28*fw,lip[1]-.17*fh]])
 points=np.vstack((foreground,background));clipped=np.clip(points,[0,0],[width-1,height-1])
 return dict(box=box.tolist(),points=clipped.tolist(),labels=[1]*5+[0]*3,n_clipped_points=int(np.any(clipped!=points,axis=1).sum()),
     note='Same anatomy rule for every prompt group; foreground points are geometric hypotheses, not verified beard pixels')
