"""Prepare a balanced human audit of C2C face-region box proposals."""
import argparse
import csv
import html
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "backprop/manifest.csv"
DEFAULT_BOXES = ROOT / "concept/category_analysis_full/bounding_boxes.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "face_sae_pilot"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def proposals(box_rows):
    by_sample = defaultdict(list)
    for row in box_rows:
        label = row["label"].strip().lower()
        if row["source"] != "objects" or label != "human face":
            continue
        priority = 0
        coords = [float(row[k]) / 512 for k in ("x1", "y1", "x2", "y2")]
        area = max(0, coords[2] - coords[0]) * max(0, coords[3] - coords[1])
        by_sample[row["sample_id"]].append((priority, area, row, coords))
    # Use the object detector's human-face boxes only; phrase grounding is not
    # part of this face-region review.
    result = {}
    for sample, candidates in by_sample.items():
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, row, coords = candidates[0]
        result[sample] = (row["source"], row["label"], coords,
                          "; ".join(f"{r['source']}:{r['label']}" for _, _, r, _ in candidates))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--per-cell", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261016)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError(f"Output exists; choose a new directory: {args.output}")
    manifest = read_csv(args.manifest)
    boxes = proposals(read_csv(args.boxes))
    groups = defaultdict(list)
    for row in manifest:
        groups[(row["race_prompt"], row["gender_prompt"])].append(row)
    rng = random.Random(args.seed)
    selected = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda r: int(r["seed"]))
        if len(candidates) < args.per_cell:
            raise ValueError(f"Only {len(candidates)} images in {key}")
        selected.extend(rng.sample(candidates, args.per_cell))
    selected.sort(key=lambda r: (r["race_prompt"], r["gender_prompt"], int(r["seed"])))
    args.output.mkdir(parents=True)
    fields = ["sample_id", "split", "race_prompt", "gender_prompt", "image", "proposal_source",
              "proposal_label", "proposal_x1", "proposal_y1", "proposal_x2", "proposal_y2",
              "other_face_part_proposals", "box_valid", "reviewed_x1", "reviewed_y1",
              "reviewed_x2", "reviewed_y2", "reviewer", "notes"]
    with (args.output / "face_region_review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in selected:
            proposal = boxes.get(row["sample_id"])
            source, label, coords, other = proposal if proposal else ("", "", ["", "", "", ""], "")
            writer.writerow(dict(sample_id=row["sample_id"], split=row["split"],
                race_prompt=row["race_prompt"], gender_prompt=row["gender_prompt"], image=row["image"],
                proposal_source=source, proposal_label=label,
                proposal_x1=coords[0], proposal_y1=coords[1], proposal_x2=coords[2], proposal_y2=coords[3],
                other_face_part_proposals=other, box_valid="", reviewed_x1="", reviewed_y1="",
                reviewed_x2="", reviewed_y2="", reviewer="", notes=""))

    cards = []
    for row in selected:
        proposal = boxes.get(row["sample_id"])
        overlay = ""
        if proposal:
            _, _, (x1, y1, x2, y2), _ = proposal
            overlay = (f'<svg viewBox="0 0 512 512" preserveAspectRatio="none"><rect '
                       f'x="{x1*512:.1f}" y="{y1*512:.1f}" width="{(x2-x1)*512:.1f}" '
                       f'height="{(y2-y1)*512:.1f}" fill="none" stroke="#00e5ff" stroke-width="3"/>'
                       f'</svg>')
        title = f"{row['sample_id']} · {row['split']} · {row['race_prompt']} {row['gender_prompt']}"
        cards.append(f'<article><div class="photo"><img loading="lazy" src="../../{html.escape(row["image"])}">'
                     f'{overlay}</div><b>{html.escape(title)}</b><br>'
                     f'<small>cyan = proposed face region; source: {html.escape(proposal[1] if proposal else "no box")}</small></article>')
    page = '''<!doctype html><meta charset="utf-8"><title>Face region box review</title>
<style>body{font:15px sans-serif;margin:20px;color:#222}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px}.photo{position:relative}.photo img{display:block;width:100%}.photo svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}article{border:1px solid #bbb;padding:8px}small{color:#555}</style>
<h1>Face region proposals: balanced audit</h1>
<p>120 images: 10 from each of 12 prompt categories, selected with a fixed seed. Cyan outlines show the proposed box. Review whether each box tightly covers the face/head region; mark <code>box_valid</code> as 1, 0, or uncertain in <code>face_region_review.csv</code>. If incorrect, enter corrected normalized coordinates from 0 to 1. Do not treat C2C proposals as verified labels.</p>
<main>''' + "\n".join(cards) + "</main>\n"
    (args.output / "gallery.html").write_text(page)
    (args.output / "settings.json").write_text(json.dumps({
        "images": len(selected), "per_prompt_category": args.per_cell,
        "seed": args.seed, "selection": "balanced random sample across 12 race/gender prompt cells",
        "proposal_source": str(args.boxes),
        "proposal_source_type": "object detection only",
        "proposal_label": "human face",
        "coordinates": "normalized x1,y1,x2,y2 in 0..1",
        "warning": "Automatic C2C proposals are unreviewed and must not be used as ground truth."
    }, indent=2) + "\n")
    print(f"Prepared {len(selected)} review rows in {args.output}")


if __name__ == "__main__":
    main()
