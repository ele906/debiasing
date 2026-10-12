"""Extract Florence-2 descriptions and grounded concepts from pilot images."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, default=HERE / 'pilot/review.csv')
    p.add_argument('--output', type=Path, default=HERE / 'extracted')
    p.add_argument('--summary-dir', type=Path, default=HERE)
    p.add_argument('--model', default='microsoft/Florence-2-base-ft')
    p.add_argument('--revision', required=True, help='Pinned Hugging Face commit SHA')
    p.add_argument('--cache', type=Path, default=HERE / 'hf_cache')
    p.add_argument('--limit', type=int)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('limit must be positive')
    with args.review.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Need nonempty unique sample IDs')
    if any(r['split'] == 'test' for r in rows):
        raise ValueError('Keep test samples reserved')
    settings = dict(model=args.model, revision=args.revision,
        review_sha256=hashlib.sha256(args.review.read_bytes()).hexdigest(),
        sample_ids=[r['sample_id'] for r in rows], tasks=['<MORE_DETAILED_CAPTION>', '<CAPTION_TO_PHRASE_GROUNDING>', '<OD>'],
        max_new_tokens=512, num_beams=3, do_sample=False, device=args.device,
        interpretation='Unreviewed detector predictions; not racism labels')
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Output settings differ: choose a fresh output')
    settings_path.write_text(json.dumps(settings, indent=2))
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoModelForCausalLM, AutoProcessor
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on a GPU node or specify --device cpu')
    dtype = torch.float16 if args.device == 'cuda' else torch.float32
    snapshot = args.cache / ('models--' + args.model.replace('/', '--')) / 'snapshots' / args.revision
    if not snapshot.is_dir():
        raise ValueError(f'Missing pinned model snapshot: {snapshot}')
    common = dict(trust_remote_code=True, local_files_only=True)
    print('Loading Florence-2 with eager attention', flush=True)
    # Transformers 4.38 treats guarded optional flash_attn imports as required.
    # Eager attention never uses them; narrowly correct this dependency scan.
    from unittest.mock import patch
    from transformers import dynamic_module_utils
    original_get_imports = dynamic_module_utils.get_imports
    def eager_imports(filename):
        imports = original_get_imports(filename)
        if Path(filename).name == 'modeling_florence2.py':
            imports = [name for name in imports if name != 'flash_attn']
        return imports
    with patch.object(dynamic_module_utils, 'get_imports', eager_imports):
        processor = AutoProcessor.from_pretrained(str(snapshot), **common)
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=dtype,
            attn_implementation='eager', **common).to(args.device).eval()

    def infer(image, task, text=''):
        inputs = processor(text=task + text, images=image, return_tensors='pt').to(args.device, dtype)
        with torch.inference_mode():
            ids = model.generate(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'],
                max_new_tokens=512, num_beams=3, do_sample=False)
        raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(raw, task=task, image_size=image.size)
        return raw, parsed[task]

    records = []
    for i, row in enumerate(rows):
        sample = row['sample_id']
        if Path(sample).name != sample:
            raise ValueError('Unsafe sample ID')
        path = args.output / f'{sample}.json'
        digest = hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()
        if path.exists():
            record = json.loads(path.read_text())
            if record['image_sha256'] != digest:
                raise ValueError('Image changed since extraction')
        else:
            with Image.open(row['image']) as source:
                image = source.convert('RGB')
            raw_caption, caption = infer(image, '<MORE_DETAILED_CAPTION>')
            raw_grounding, grounding = infer(image, '<CAPTION_TO_PHRASE_GROUNDING>', caption)
            raw_od, objects = infer(image, '<OD>')
            record = dict(sample_id=sample, image=row['image'], image_sha256=digest,
                caption=caption, grounding=grounding, objects=objects,
                raw=dict(caption=raw_caption, grounding=raw_grounding, objects=raw_od))
            path.write_text(json.dumps(record, indent=2))
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            for label, box in zip(grounding['labels'], grounding['bboxes']):
                draw.rectangle(box, outline='red', width=2)
                draw.text((box[0], box[1]), label, fill='red', stroke_width=1, stroke_fill='white')
            overlay.save(args.output / f'{sample}_grounded.jpg')
        records.append(record)
        print(f'{i+1}/{len(rows)} {sample}: {record["caption"]}', flush=True)
    counts = Counter()
    table = []
    for row, record in zip(rows, records):
        concepts = sorted({str(s).strip().lower() for source in ['grounding', 'objects'] for s in record[source]['labels'] if str(s).strip()})
        counts.update(concepts)
        table.append(dict(sample_id=row['sample_id'], split=row['split'], race_prompt=row['race_prompt'],
            gender_prompt=row['gender_prompt'], image=row['image'], features=row['features'],
            caption=record['caption'], detected_concepts='; '.join(concepts), concepts='',
            harmful_depiction='', harm_reason='', reviewer='', notes=''))
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    with (args.summary_dir / 'concept_predictions.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, lineterminator='\n', fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    with (args.summary_dir / 'concept_frequencies.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['detected_concept', 'image_count', 'image_fraction'])
        w.writerows((c, n, n / len(records)) for c, n in counts.most_common())
    report = ['# Extracted concept candidates', '', 'Florence-2 predictions, not validated concepts or racism labels.', '', f'Images: {len(records)}', '', '## Most frequent detector phrases', '']
    report += [f'- {c}: {n}/{len(records)} images' for c, n in counts.most_common(30)]
    report += ['', '## Example descriptions', '']
    report += [f'- {r["sample_id"]}: {r["caption"]}' for r in records[:12]]
    report += ['', 'Review concept_predictions.csv and copy corrected labels into its concepts column before using pilot.py rank. Do not rank the unreviewed detected_concepts column. Phrase synonyms are not merged automatically.']
    (args.summary_dir / 'REPORT.md').write_text('\n'.join(report))
    print('Extraction complete. Human review and SAE mapping remain.', flush=True)


if __name__ == '__main__':
    main()
