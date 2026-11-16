"""Publish-only derivatives of saved photographs; originals remain untouched."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
for d in ('gallery','features','results','code'):(out/d).mkdir()
cfg=json.loads((run/'config/config.json').read_text());features=cfg['features']
for f in features:
 source=run/'gallery'/f'feature_{f}.png';shutil.copy2(source,out/'gallery'/source.name)
 with Image.open(source) as sheet:
  panel=Image.new('RGB',(1280,669),'white');panel.paste(sheet.crop((0,0,1280,65)),(0,0))
  for j,row in enumerate((0,4)):panel.paste(sheet.crop((0,65+row*302,1280,65+(row+1)*302)),(0,65+j*302))
  panel.save(out/'gallery'/f'paired_{f}.png')
  panel.save(run/'gallery'/f'paired_{f}.png')
 text=f'''# Feature {f}: existing photos\n\nSame saved prompt and seed30 across columns: **unchanged → multiply×7 → add+10 → add+20 → add+50**. Each row is a prompt group. These are direct viewing derivatives of old PNGs, not new generations. Originals remain at512×512 in the experiment archive. Multipliers and additive constants are different units; these columns are not perturbation-matched.\n\n[Back to all ten features](../README.md)\n\n![All twelve prompt groups](../gallery/feature_{f}.png)\n'''
 (out/'features'/f'feature_{f}.md').write_text(text)
text='''# Gallery: the old ten-feature intervention photos\n\n**These photos already existed. No new generations were run.** The gallery arranges saved results from the face-fine-tuned mid.0 SAE (`cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6`) and the same frozen SDXL Turbo generator.\n\nEvery preview uses **seed30**, chosen before reviewing effects. Columns are **unchanged, multiplication×7, addition+10,+20,+50**. All ten features and all12 prompt groups are included. Click a feature page for all groups. The two-row previews below show Black-man and East-Asian-man prompt contexts. Prompt categories are conditioning labels, not ground-truth identities.\n\nThese multiplier/additive columns are **not magnitude-matched**. Quantitative matched-condition summaries and the README/LaTeX research report are provided separately. A photo that looks similar at thumbnail size is not proof of zero effect.\n\n'''
for f in features:text+=f'## [Feature {f}: all twelve groups](features/feature_{f}.md)\n\n![Saved feature {f} images](gallery/paired_{f}.png)\n\n'
text+='''## What these images can and cannot tell us\n\nWe have not established reliable single-coordinate semantic control for these features. That is different from saying all edits have no effect. Pixel change, demographic prediction, facial-hair change and image quality are separate outcomes.\n\nWeak or inconsistent effects do not show that “race itself” is the cause. Prompt wording, distributed activations, spatial structure, feature interactions and the measurement being used remain possible explanations. The tested feature edits are interventions on a model representation, not on an intrinsic human race variable.\n\nOriginals are untouched. The gallery consists of256px viewing derivatives; all512px originals, latent archives and previous results remain in `/n/fs/goose/experiments/mid0_sae_selectivity/additive_20261114_174738_954258/` and the matching multiplicative sweep. [Every gallery photo's source path and hash](results/gallery_sources.csv) is recorded.\n'''
(out/'README.md').write_text(text)
shutil.copy2(run/'results/gallery_sources.csv',out/'results/gallery_sources.csv')
for source in Path(__file__).parent.glob('*'):
 if source.suffix in ('.py','.sbatch'):shutil.copy2(source,out/'code'/source.name)
print(out)
