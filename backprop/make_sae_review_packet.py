"""Build a small human-review packet for an SAE concept pilot."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'backprop'
OUT = DATA / 'sae_review_60'
RACES = ['black', 'white', 'east asian', 'middle eastern', 'indian', 'hispanic']
GENDERS = ['man', 'woman']


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    if OUT.exists():
        raise FileExistsError(f'{OUT} already exists; preserve it and choose a new output in code to make another packet')

    source = read_csv(DATA / 'manifest.csv')
    rng = np.random.default_rng(args.seed)
    selected = []
    for race in RACES:
        for gender in GENDERS:
            group = [r for r in source if r['race_prompt'] == race and r['gender_prompt'] == gender]
            train = [r for r in group if r['split'] == 'train']
            validation = [r for r in group if r['split'] == 'validation']
            if len(train) < 4 or not validation:
                raise ValueError(f'Not enough train/validation samples for {race} {gender}')
            chosen = [(r, 'train') for r in rng.choice(train, 4, replace=False)]
            chosen += [(r, 'validation') for r in rng.choice(validation, 1, replace=False)]
            for row, split in chosen:
                selected.append(dict(sample_id=row['sample_id'], race_prompt=race,
                    gender_prompt=gender, seed=row['seed'], split=split,
                    source_image=str((DATA / row['image']).resolve()),
                    feature_file=str((DATA / row['features']).resolve()),
                    turban_present='', flower_present='', kimono_present='',
                    reviewer='', notes=''))

    OUT.mkdir(parents=True)
    (OUT / 'contact_sheets').mkdir()
    write_csv(OUT / 'annotations.csv', selected)
    (OUT / 'settings.json').write_text(json.dumps(dict(samples=len(selected),
        samples_per_group=5, train_per_group=4, validation_per_group=1,
        test_samples=0, selection_seed=args.seed,
        selection='Uniform random sample within each race/gender group and source split; no concept labels used for selection',
        labels=['turban_present', 'flower_present', 'kimono_present'],
        label_values='Use 1=present, 0=absent, uncertain=uncertain; blank=not reviewed'), indent=2) + '\n')

    for race in RACES:
        for gender in GENDERS:
            group = [r for r in selected if r['race_prompt'] == race and r['gender_prompt'] == gender]
            sheet = Image.new('RGB', (3 * 256, 2 * 292), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, row in enumerate(group):
                x, y = (i % 3) * 256, (i // 3) * 292
                image = Image.open(row['source_image']).convert('RGB')
                sheet.paste(image.resize((256, 256)), (x, y + 24))
                draw.text((x + 3, y + 3), f'{row["sample_id"]} [{row["split"]}]', fill='black')
            name = f'{race.replace(" ", "_")}_{gender}.jpg'
            sheet.save(OUT / 'contact_sheets' / name, quality=90)

    lines = ['# 60-image SAE review packet', '',
        'Five randomly selected portraits per race/gender prompt group: four train and one validation image per group. The held-out test split is untouched. Selection did not use concept labels or SAE feature values.', '',
        'Open each group contact sheet, then record labels in `annotations.csv`: `1` = visible, `0` = absent, `uncertain` = cannot tell, blank = not reviewed. Label the visible object only; flower prints in clothing/background do not count as a flower object. Count a clearly visible kimono garment, and a clearly visible turban headwrap; do not infer an item from the person’s prompt category. Clothing and accessories are presence labels, not harm labels.', '',
        'This is a small pilot. A concept may have too few positive examples, especially in one split. If so, do not train a probe from it; collect more reviewed examples.']
    for race in RACES:
        for gender in GENDERS:
            name = f'{race.replace(" ", "_")}_{gender}.jpg'
            lines += ['', f'## {race} {gender}', '', f'![{race} {gender} sample contact sheet](contact_sheets/{name})']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(f'Created {len(selected)} review rows and 12 contact sheets at {OUT}')


if __name__ == '__main__':
    main()
