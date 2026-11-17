"""Preserve sub-threshold parser onsets separately from 1% threshold crossings."""
import argparse,collections
from pathlib import Path
from prepare import read,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();eligible=collections.defaultdict(set);positive=collections.defaultdict(set);maximum=collections.defaultdict(int)
 for r in read(run/'results/per_image_scores.csv'):
  if r['method']=='additive' and int(r['baseline_beard_pixels'])==0:
   f=r['feature_id'];key=(r['prompt_group'],r['seed']);eligible[f].add(key)
   if int(r['beard_pixels'])>0:positive[f].add(key);maximum[f]=max(maximum[f],int(r['beard_pixels']))
 write(run/'results/any_pixel_initiation.csv',[dict(feature_id=f,n_unique_parser_zero_contexts=len(eligible[f]),n_contexts_with_any_positive_beard_pixel=len(positive[f]),max_beard_pixels_from_parser_zero_context=maximum[f],interpretation='Any positive argmax pixel, not validated beard presence; repeated offsets count once per context') for f in sorted(eligible,key=int)])
 page=run/'README.md';text=page.read_text();note='Single-pixel sensitivity: `results/any_pixel_initiation.csv` retains any positive class14 pixels arising from exact parser-zero baselines. Small predictions do occur below the1% criterion, so absence of1% crossings must not be read as absence of every predicted beard pixel or proof that addition cannot initiate hair.\n\n'
 if note not in text:text=text.replace('## Figures\n\n',note+'## Figures\n\n')
 page.write_text(text)
if __name__=='__main__':main()
