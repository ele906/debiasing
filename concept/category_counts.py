"""Count extracted concepts by prompt category and draw all detected boxes."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
from compare_sae import PATTERNS

HERE=Path(__file__).resolve().parent


def write(path, rows, fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,default=HERE/'concept_predictions.csv')
    p.add_argument('--records-dir', type=Path, default=HERE/'extracted')
    p.add_argument('--sheet-limit', type=int, default=6, help='Examples shown per category; all images receive full-size boxes')
    p.add_argument('--output',type=Path,default=HERE/'category_analysis')
    args=p.parse_args()
    if args.sheet_limit < 1: p.error('sheet-limit must be positive')
    if args.output.exists():raise ValueError('Choose a new output directory')
    with args.predictions.open() as f:rows=list(csv.DictReader(f))
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Unique nonempty samples required')
    args.output.mkdir(parents=True)
    groups={}
    for row in rows:groups.setdefault((row['race_prompt'],row['gender_prompt']),[]).append(row)
    def concepts(row):
        labels={s.strip().lower() for s in row['detected_concepts'].split(';') if s.strip()}
        return {c for c,pattern in PATTERNS.items() if any(re.search(pattern,s) for s in labels)}
    normalized={r['sample_id']:concepts(r) for r in rows}
    counts,rates,raw=[],[],[]
    vocabulary=sorted({s.strip().lower() for r in rows for s in r['detected_concepts'].split(';') if s.strip()})
    for (race,gender),samples in groups.items():
        n=len(samples)
        count=Counter(c for r in samples for c in normalized[r['sample_id']])
        base=dict(category=f'{race} {gender}',race_prompt=race,gender_prompt=gender,images=n)
        counts.append(dict(**base,**{c:count[c] for c in PATTERNS}))
        rates.append(dict(**base,**{c:count[c]/n for c in PATTERNS}))
        tally=Counter(c for r in samples for c in {s.strip().lower() for s in r['detected_concepts'].split(';') if s.strip()})
        raw.append(dict(**base,**{c:tally[c] for c in vocabulary}))
    fields=['category','race_prompt','gender_prompt','images']
    write(args.output/'counts_by_category.csv',counts,fields+list(PATTERNS))
    write(args.output/'rates_by_category.csv',rates,fields+list(PATTERNS))
    write(args.output/'raw_phrase_counts_by_category.csv',raw,fields+vocabulary)
    pooled=[]
    for race in dict.fromkeys(r['race_prompt'] for r in rows):
        samples=[r for r in rows if r['race_prompt']==race]
        tally=Counter(c for r in samples for c in normalized[r['sample_id']])
        pooled.append(dict(race_prompt=race,images=len(samples),**{c:tally[c] for c in PATTERNS}))
    write(args.output/'counts_by_race.csv',pooled,['race_prompt','images']+list(PATTERNS))
    report=['# Concept counts by prompt category','','Counts are images containing a detector label, not numbers of boxes. Repeated labels in an image count once. Targets are unreviewed predictions; missing labels may be detector misses. Clothing and hairstyle presence is not a racism label.','',f'This extraction has {len(groups)} categories, {len(rows)} images total. The source dataset defines six race prompts and two gender prompts (12 categories), not 16. Counts use every supplied image; denominators appear in the table. Test labels, when supplied, are descriptive only and must be excluded from SAE feature selection.','']
    for chunk in [list(PATTERNS)[:7],list(PATTERNS)[7:]]:
        report += ['| Category | Images | '+' | '.join(chunk)+' |','|---|---:|'+'---:|'*len(chunk)]
        for row in counts:report.append('| '+row['category']+' | '+str(row['images'])+' | '+' | '.join(str(row[c]) for c in chunk)+' |')
        report.append('')
    report += ['## How labels are matched','','The normalized table uses these exact detector-phrase patterns (caption-only mentions are excluded):','']
    report += [f'- `{c}`: `{pattern}`' for c,pattern in PATTERNS.items()]
    report += ['', 'Raw phrases are preserved in raw_phrase_counts_by_category.csv. Rates are fractions from 0 to 1 in rates_by_category.csv. The pilot is small and these are descriptive counts, not population estimates or significance tests.', '', '## Bounding boxes', '', 'Category JPG sheets below show all unique phrase-grounding and object-detection boxes for each image. Numbered boxes correspond to the legend below each tile. Blue: phrase grounding; red: object detection. Both sources can disagree or miss concepts. Malformed reversed-coordinate boxes are preserved in the raw JSON but omitted from these drawings. Some large boxes label whole people or abstract descriptions rather than distinct objects. These boxes are not SAE heatmaps.', '']
    report += [f'Category sheets show the first {args.sheet_limit} examples per category. Individual boxed images exist under boxes/ for every supplied sample.', '']
    bbox_rows=[]
    (args.output/'boxes').mkdir()
    for (race,gender),samples in groups.items():
        tiles=[]
        for row in samples:
            sample=row['sample_id']
            record=json.loads((args.records_dir/f'{sample}.json').read_text())
            with Image.open(row['image']) as src:image=src.convert('RGB')
            draw=ImageDraw.Draw(image)
            seen=set();legend=[]
            for source,color in [('grounding','blue'),('objects','red')]:
                result=record[source]
                if len(result['labels'])!=len(result['bboxes']):raise ValueError('Label/box mismatch')
                for label,box in zip(result['labels'],result['bboxes']):
                    key=(str(label).lower().strip(),tuple(round(float(v),1) for v in box))
                    if key in seen:continue
                    seen.add(key)
                    if len(box)!=4 or box[2]<box[0] or box[3]<box[1]:
                        print(f'Skipping malformed box: {sample} {source} {label}',flush=True)
                        continue
                    number=len(legend)+1
                    draw.rectangle(box,outline=color,width=2)
                    draw.text((box[0],box[1]),str(number),fill=color,stroke_width=1,stroke_fill='white')
                    legend.append((number,source,str(label),color))
                    bbox_rows.append(dict(sample_id=sample,category=f'{race} {gender}',number=number,source=source,label=label,x1=box[0],y1=box[1],x2=box[2],y2=box[3]))
            # Full-size per-image render keeps boxes legible. Category sheets are smaller.
            panel=Image.new('RGB',(512,548+16*len(legend)),'white')
            panel.paste(image,(0,28));pd=ImageDraw.Draw(panel)
            pd.text((4,5),sample,fill='black')
            for i,(number,source,label,color) in enumerate(legend):pd.text((4,544+16*i),f'{number}: {label} ({source})',fill=color)
            panel.save(args.output/'boxes'/f'{sample}.jpg',quality=90)
            tile=Image.new('RGB',(384,630),'white')
            tile.paste(image.resize((384,384)),(0,28));td=ImageDraw.Draw(tile)
            td.text((4,6),sample,fill='black')
            for i,(number,source,label,color) in enumerate(legend):
                if 418+13*i>615:break
                td.text((4,418+13*i),f'{number}: {label[:45]}',fill=color)
            if len(tiles)<args.sheet_limit: tiles.append(tile)
        name=race.replace(' ','_')+'_'+gender
        sheet=Image.new('RGB',(3*384,((len(tiles)+2)//3)*630),'white')
        for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*384,(i//3)*630))
        sheet.save(args.output/f'{name}_boxes.jpg',quality=90)
        report += [f'### {race} {gender}','','!['+name+']('+name+'_boxes.jpg)','']
    write(args.output/'bounding_boxes.csv',bbox_rows,['sample_id','category','number','source','label','x1','y1','x2','y2'])
    (args.output/'REPORT.md').write_text('\n'.join(report).rstrip()+'\n')
    (args.output/'settings.json').write_text(json.dumps(dict(predictions=str(args.predictions),patterns=PATTERNS,samples=len(rows),categories=len(groups),label_source='Detector grounded/object phrases; not human labels'),indent=2)+'\n')
    print(f'{len(groups)} category tables; {len(rows)} boxed images; {len(bbox_rows)} boxes',flush=True)
    for row in counts:print(row['category'], 'turban=',row['turban'],'headscarf=',row['headscarf'],'of',row['images'],flush=True)


if __name__=='__main__':main()
