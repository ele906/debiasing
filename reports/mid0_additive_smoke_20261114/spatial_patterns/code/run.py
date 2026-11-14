import csv,hashlib,json,os
from pathlib import Path
ROOT=Path(os.environ.get("SAE_PROJECT_ROOT","/n/fs/goose"))
BLOCK="unet.mid_block.attentions.0"

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def events(run):
    path = run / 'logs/events.jsonl'
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed write is retained as evidence; subsequent writes start a new line.
            continue
    return result

def record(run, row):
    with (run / 'logs/events.jsonl').open('a') as f:
        f.write('\n' + json.dumps(row) + '\n')
        f.flush()
        os.fsync(f.fileno())

def inventory(run):
    files = [dict(path=str(p.relative_to(run)), bytes=p.stat().st_size, sha256=digest(p))
             for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ('output_manifest.json', 'run.lock')]
    atomic_json(run / 'output_manifest.json', files)

def index(run):
    latest = {}
    for e in events(run):
        latest[e['image_id']] = e
    csv_write(run / 'results/all_generations.csv', list(latest.values()))
    return latest

