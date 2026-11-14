"""Read cached weight files sequentially before loading mapped tensors."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
cache=ROOT/'ReNO/hf_cache'
paths=list((cache/'models--stabilityai--sdxl-turbo').glob('snapshots/*/*/*.fp16.safetensors'))
paths+=list((cache/'models--madebyollin--sdxl-vae-fp16-fix').glob('snapshots/*/*.safetensors'))
seen=set()
for path in paths:
    resolved=path.resolve()
    if resolved in seen:continue
    seen.add(resolved)
    size=0
    with path.open('rb') as f:
        while True:
            chunk=f.read(16*1024*1024)
            if not chunk:break
            size+=len(chunk)
    print(f'Prefetched {path.parent.name}/{path.name}: {size/1024**2:.1f} MiB',flush=True)
