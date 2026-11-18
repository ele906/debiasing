"""Direct RGB pixel audit for existing slide21 images, without model inference."""
import datetime as dt,itertools,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from prepare import BASE,read,write,sha

def main():
 source=BASE/'ten_feature_review_20261116_230430_705065/results/per_image_pixel_results.csv';rows=[r for r in read(source) if r['feature_id']=='4228' and r['seed']=='30' and r['strength_label'] in ['fixed_10','fixed_20','fixed_50']]
 run=BASE/('slide21_pixel_audit_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
 for d in ['figures','differences','results','config/code']:(run/d).mkdir(parents=True,exist_ok=True)
 results=[];examples=[];font=lambda n:ImageFont.truetype('DejaVuSans.ttf',n)
 for group in sorted({r['group'] for r in rows}):
  rr=sorted([r for r in rows if r['group']==group],key=lambda r:float(r['additive_offset']));paths=[Path(rr[0]['baseline_image_path'])]+[Path(r['image_path']) for r in rr];arrays=[]
  for p in paths:
   with Image.open(p) as im:assert im.size==(512,512);arrays.append(np.asarray(im.convert('RGB')).copy())
  for i,j in itertools.combinations(range(4),2):
   delta=arrays[j].astype(np.int16)-arrays[i].astype(np.int16);absolute=np.abs(delta);changed=absolute.max(2);results.append(dict(prompt_group=group,seed=30,reference_beta=[0,10,20,50][i],edited_beta=[0,10,20,50][j],pixel_mae=float(absolute.mean()),pixel_rmse=float(np.sqrt((delta.astype(np.float64)**2).mean())),max_channel_change=int(absolute.max()),changed_pixels=int((changed>0).sum()),changed_pixel_percent=100*float((changed>0).mean()),pixels_max_channel_change_at_least10_percent=100*float((changed>=10).mean()),pixels_identical=bool(np.array_equal(arrays[i],arrays[j])),reference_path=str(paths[i]),edited_path=str(paths[j]),reference_sha256=sha(paths[i]),edited_sha256=sha(paths[j])))
  sheet=Image.new('RGB',(1024,660),'white');draw=ImageDraw.Draw(sheet);draw.text((8,5),group.replace('_',' ').title()+' | seed30',font=font(20),fill='black')
  for col,beta in enumerate([0,10,20,50]):
   draw.text((col*256+8,36),'Baseline' if beta==0 else f'Add +{beta}',font=font(17),fill='black');sheet.paste(Image.fromarray(arrays[col]).resize((256,256)),(col*256,65))
   delta=arrays[col].astype(np.int16)-arrays[0].astype(np.int16);absolute=np.abs(delta);magnitude=absolute.mean(2)
   if beta:
    np.savez_compressed(run/'differences'/f'{group}_beta{beta}.npz',signed_rgb_difference=delta,absolute_rgb_difference=absolute)
    # Display absolute RGB difference with a fixed4x amplification across every sheet.
    display=np.minimum(absolute.astype(np.int32)*4,255).astype(np.uint8);r=next(r for r in results if r['prompt_group']==group and r['reference_beta']==0 and r['edited_beta']==beta);caption=f'Mean |ΔRGB| = {r["pixel_mae"]:.3f}'
   else:display=np.zeros_like(arrays[0]);caption='Zero reference difference'
   draw.text((col*256+8,336),caption,font=font(14),fill='black');sheet.paste(Image.fromarray(display).resize((256,256)),(col*256,365))
  draw.text((8,634),'Bottom row: absolute RGB difference from baseline, amplified x4 for visibility (display only).',font=font(13),fill='black');sheet.save(run/'figures'/(group+'.png'))
  coords=[('fixed_center',256,256)];delta=np.abs(arrays[3].astype(np.int16)-arrays[0].astype(np.int16)).sum(2);y,x=np.unravel_index(delta.argmax(),delta.shape);coords.append(('largest_total_RGB_change_for_beta50',int(x),int(y)))
  for rule,x,y in coords:
   examples.append(dict(prompt_group=group,selection_rule=rule,x=x,y=y,baseline_RGB=json.dumps(arrays[0][y,x].tolist()),add10_RGB=json.dumps(arrays[1][y,x].tolist()),add20_RGB=json.dumps(arrays[2][y,x].tolist()),add50_RGB=json.dumps(arrays[3][y,x].tolist())))
 assert len(results)==72 and not any(r['pixels_identical'] for r in results)
 write(run/'results/pairwise_pixel_metrics.csv',results);write(run/'results/actual_pixel_examples.csv',examples)
 (run/'config/config.json').write_text(json.dumps(dict(feature=4228,seed=30,groups=12,pairwise_comparisons=72,source=str(source),source_sha256=sha(source),pixel_units='RGB integer intensity0–255',difference_visualization='absolute RGB differences times4, capped255 for viewing only; numerical metrics and NPZ differences are not amplified or clipped',models_run=0),indent=2))
 for p in [Path(__file__),Path(__file__).parent/'prepare.py']:shutil.copy2(p,run/'config/code'/p.name)
 text='# Slide21: direct pixel differences\n\nFeature4228, seed30, all12 prompt groups. Each group’s baseline/+10/+20/+50 columns are pairwise pixel-different:72/72 comparisons. These differences do not establish meaningful semantic or beard changes.\n\nMAE = mean(abs(editedRGB − referenceRGB)) across all512×512×3 channel values, in0–255 intensity units. Changed-pixel percentage counts positions where at least one channel differs. The CSV also records changes of at least10 intensity units.\n\n| Prompt group | +10 vs baseline MAE | +20 MAE | +50 MAE |\n|---|---:|---:|---:|\n'
 for g in sorted({r['prompt_group'] for r in results}):text+='| '+g+' | '+' | '.join(f"{next(r for r in results if r['prompt_group']==g and r['reference_beta']==0 and r['edited_beta']==b)['pixel_mae']:.3f}" for b in [10,20,50])+' |\n'
 text+='\n## Difference previews\n\nTop rows show saved photos. Bottom rows show absolute RGB differences from baseline, amplified4× for visibility and capped at255. This amplification is for display only. Full unamplified signed and absolute differences are retained losslessly in NPZ on the research host.\n\n'
 for g in sorted({r['prompt_group'] for r in results}):text+=f'### {g}\n\n![Pixel differences](figures/{g}.png)\n\n'
 text+='results/actual_pixel_examples.csv lists actual RGB values at the same coordinate in all four conditions. It includes a fixed center pixel and an explicitly selected maximum-difference pixel; the latter is not representative. No image generation, model inference or source-image modification was performed.\n'
 (run/'README.md').write_text(text);write(run/'output_manifest.csv',[dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='output_manifest.csv']);print('RUN_DIRECTORY',run,flush=True)
 for r in results:
  if r['reference_beta']==0:print(r['prompt_group'],r['edited_beta'],round(r['pixel_mae'],3),round(r['changed_pixel_percent'],2),flush=True)
 for r in examples:
  if r['prompt_group']=='indian_man' and r['selection_rule']=='fixed_center':print('EXAMPLE',json.dumps(r),flush=True)
if __name__=='__main__':main()
