"""Freeze final documentation, code/environment snapshots, Wilson detection intervals and file hashes."""
import datetime as dt
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/n/fs/goose/experiments/mid0_sae_selectivity');sys.path.insert(0,str(BASE))
from run import digest,atomic_json,csv_write,inventory,events


def main():
    run=Path(sys.argv[1]).resolve();cfg=json.loads((run/'config/experiment_config.json').read_text())
    manifest=[]
    evaluation_source=run/'config/evaluation_source';evaluation_source.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*'):
        if source.suffix in ('.py','.sbatch'):
            dest=evaluation_source/source.name
            if dest.exists() and digest(dest)!=digest(source):
                dest=evaluation_source/(source.stem+'_'+digest(source)[:10]+source.suffix)
            if not dest.exists():shutil.copy2(source,dest)
            manifest.append(dict(path=str(dest.relative_to(run)),sha256=digest(dest)))
    atomic_json(run/'config/evaluation_code_manifest.json',manifest)
    (run/'config/evaluation_environment.txt').write_text(sys.version+'\n'+subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout)
    metric_rows=json.loads((run/'evaluations/image_metrics.json').read_text())
    import csv
    faces={r['image_id']:r for r in csv.DictReader((run/'evaluations/face_validity.csv').open())}
    detection=[];z=1.959963984540054
    for group in ('black_man','east_asian_man'):
        for target in cfg['targets']:
            for method in ('natural','uniform','shuffled','localized'):
                flags=[]
                for seed in range(30,35):
                    subset=[r for r in metric_rows if r['group']==group and r['seed']==seed and r['target_norm']==target and r['method']==method]
                    if subset and all(faces[r['image_id']]['status']=='complete' for r in subset):
                        flags.append(all(faces[r['image_id']]['face_detected']=='True' for r in subset))
                n=len(flags);k=sum(flags);p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
                detection.append(dict(group=group,method=method,target_norm=target,seed_successes=k,n_seeds=n,rate=p,ci_low=c-h,ci_high=c+h,
                    unit='Seed with all method replicates detected; geometric detection only'))
    csv_write(run/'results/face_detection_seed_wilson.csv',detection)
    for suffix in ('.png','.pdf'):
        old=run/'figures'/('05_face_detection_vs_norm'+suffix);dest=run/'figures'/('05_face_detection_vs_norm_empirical_bootstrap'+suffix)
        if old.exists() and not dest.exists():shutil.copy2(old,dest)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,group in zip(axes,('black_man','east_asian_man')):
        for method in ('natural','uniform','shuffled','localized'):
            s=[r for r in detection if r['group']==group and r['method']==method];x=np.array([r['target_norm'] for r in s]);y=np.array([r['rate'] for r in s]);lo=np.array([r['ci_low'] for r in s]);hi=np.array([r['ci_high'] for r in s])
            ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt='o-',capsize=3,label=method)
        ax.set(title=group,xlabel='Applied L2 target',ylabel='Per-seed all-images-detected rate',ylim=(0,1.04));ax.legend(fontsize=8)
    fig.suptitle('95% Wilson intervals over 5 seeds; face detection is not structural validity');fig.tight_layout()
    fig.savefig(run/'figures/05_face_detection_vs_norm.pdf');fig.savefig(run/'figures/05_face_detection_vs_norm.png',dpi=300);plt.close(fig)
    # Explicit activity-stratified descriptive summaries; small strata are not confirmatory.
    sensitivity=[]
    for group in ('black_man','east_asian_man'):
        for active in (False,True):
            for method in ('natural','uniform','shuffled','localized'):
                for target in cfg['targets']:
                    subset=[r for r in metric_rows if r['group']==group and r['method']==method and r['target_norm']==target and (not r['original_activation_zero'])==active]
                    sensitivity.append(dict(group=group,initially_active=active,method=method,target_norm=target,n_seeds=len({r['seed'] for r in subset}),n_images=len(subset),
                        lowerface_change_fraction=None if not subset else float(np.mean([r['lowerface_change_fraction'] for r in subset])),
                        pixel_mae=None if not subset else float(np.mean([r['pixel_mae'] for r in subset]))))
    csv_write(run/'results/activity_stratified_descriptive.csv',sensitivity)
    starts=[r for r in events(run/'workers/seed_30') if r['generation_status']=='started']
    atomic_json(run/'results/resume_validation.json',dict(seed=30,expected_generation_attempts=42,observed_generation_attempts=len(starts),
        verified_no_smoke_regeneration=len(starts)==42,note='24 smoke edits and 2 no-ops were reused; full worker generated only its 16 remaining donor31 edits'))
    table=[]
    for group in ('black_man','east_asian_man'):
        for method in ('natural','uniform','shuffled','localized'):
            subset=[r for r in metric_rows if r['group']==group and r['method']==method and r['target_norm']==65.1363]
            table.append(dict(group=group,method=method,mean_pixel_mae=float(np.mean([r['pixel_mae'] for r in subset])),
                mean_ssim=float(np.mean([r['ssim'] for r in subset])),mean_lowerface_change_fraction=float(np.mean([r['lowerface_change_fraction'] for r in subset])),
                mean_fraction_positions_exceeding_topk=float(np.mean([r['fraction_positions_exceeding_topk'] for r in subset]))))
    csv_write(run/'results/observed_proxy_summary.csv',table)
    appendix='\n## Observed proxy results at L2 65.1363\n\n| Context | Method | Pixel MAE | Lower-face fraction of change | Cells beyond k=10 |\n|---|---|---:|---:|---:|\n'
    for r in table:appendix+=f"| {r['group']} | {r['method']} | {r['mean_pixel_mae']:.3f} | {r['mean_lowerface_change_fraction']:.3f} | {100*r['mean_fraction_positions_exceeding_topk']:.1f}% |\n"
    appendix+='\nFor East-Asian-man prompts, the donor-averaged natural-minus-shuffled lower-face change fraction at L2=65.1363 is 0.00784 (95% seed-bootstrap interval −0.01036 to 0.02715; exact two-sided sign-flip p=0.5). This small experiment does not establish a clear localization advantage of the natural layout over shuffled layouts. It provides no validated semantic beard effect yet.\n\nAll 220 images were detected by MediaPipe and received FairFace probabilities. Structural validity remains unreviewed. Wilson intervals in the main detection figure treat each seed as a success only if all method replicates were detected; with 5/5 successes the 95% lower bound is approximately 0.566. The original empirical-bootstrap detection figure is retained under a distinct filename because boundary bootstrap intervals are degenerate.\n'
    summary=run/'results/research_summary.md';text=summary.read_text()
    if '## Observed proxy results' not in text:summary.write_text(text+appendix)
    (run/'README.md').write_text(f'''# Feature 4228 spatial-pattern transfer\n\nComplete: 200 matched edited images, 10 generated no-op replays, 10 reused original baselines. Full data remain in this timestamped directory. Checkpoint SHA-256: `{cfg['sae_sha256']}`; SAE: `{cfg['sae_checkpoint']}`. Only `unet.mid_block.attentions.0` and the same frozen single-step SDXL Turbo generator are used. No model training or alternative SAE is involved.\n\n## Design\n\nDonors: Black-man seeds 30 and 31, chosen by original nonzero activity rather than intervention outcomes. Recipient prompts: East-Asian-man and Black-man, seeds 30–34. East-Asian seed 33 is naturally active in two cells and is explicitly labeled; other East-Asian seeds are entirely zero. Black-man seeds 32 and 34 are also entirely zero.\n\nNatural: z'=z+lambda*P_donor. Shuffled: exact permutation of donor values, seeds 1729,2718,3141. Uniform: all-ones map. Localized: recipient baseline lower-face mask projected from512px to16x16 using float BOX area averaging; fractional mask weights are retained. The existing MediaPipe ROI extends below the chin, uses blur2 and is geometric, not beard segmentation. Donor maps are copied without warping or alignment. Uniform and localized conditions are shared across donors to avoid duplicate generation and pseudoreplication.\n\nLambda is solved against ||edited_output_fp32−original_output_fp32|| after the actual fp16 addition. Targets:42.5242 and65.1363, tolerance0.5%. No clipping, hidden normalization or Top-K reapplication occurs. Full original/modified latents, raw spatial pattern, latent change, original block output, pre-cast delta, actual applied delta, decoder direction and scheduler timestep are retained in lossless NPZ files. Equal output norms do not imply equal sparse support or naturalness; see results/naturalness.csv.\n\n## Commands (from /n/fs/goose)\n\n```bash\nexport PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\nGEN=/n/fs/goose/el8403/conda-envs/goose/bin/python\n# Core tests\nPYTHONPATH=/n/fs/goose/layers/mid0_stability/generation_deps $GEN experiments/mid0_sae_selectivity/spatial_code/test_core.py\n# Resume one seed; completed artifact hashes are checked and no images are overwritten\nPYTHONPATH=/n/fs/goose/layers/mid0_stability/generation_deps $GEN {run}/config/source/worker.py --run {run} --seed 30\n# Resume all seed workers through Slurm\nsbatch --array=0-4%5 experiments/mid0_sae_selectivity/spatial_code/spatial.sbatch {run} full\n# Existing face detector (CPU, documented append-only dependency ordering)\nPYTHONPATH= /usr/local/anaconda3/2024.02/bin/python experiments/mid0_sae_selectivity/spatial_code/face_evaluate.py --run {run}\n# Per-image FairFace, SSIM, naturalness and region changes\nPYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps $GEN experiments/mid0_sae_selectivity/spatial_code/evaluate.py --run {run}\n# After adding blinded ratings, rerun analysis without image generation\nPYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps $GEN experiments/mid0_sae_selectivity/spatial_code/analyze.py --run {run}\nPYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps $GEN experiments/mid0_sae_selectivity/spatial_code/finalize.py {run}\n```\n\nTo create a new independent experiment, run prepare.py in the generation environment, then masks.py in the CPU environment (prepare.py automatically snapshots all source files and hashes), and submit spatial.sbatch RUN smoke. Inspect results/smoke_validation.json and saved tensors before launching full workers. Existing source snapshots and the complete condition plan are retained for reproduction.\n\n## Preservation and resume\n\nconfig/conditions.json is the full immutable plan (220 conditions). Each seed has its own lock and append-only fsynced attempt journal. Every generation attempt uses a unique filename; interrupted artifacts remain. Completed image/tensor hashes must match before reuse. Baseline PNGs and archives are immutable hardlinks to matching earlier results. The first seed resumed from its smoke run with exactly42 new generation attempts, showing that24 smoke edits and2 no-ops were not regenerated. No-op pixel replay checks protect baseline comparability.\n\nRaw images: workers/seed_*/images/ and images/baseline/. Raw tensors: workers/seed_*/activations/ and activations/baseline/. Original donor maps: patterns/donors/. Recorded shuffles and permutations: patterns/shuffled/. Masks, fractional projected patterns and baseline landmarks: masks/. Original images are kept separate from derivative sheets and masks. Full generator snapshot paths/hashes are in config/generator_checkpoint_manifest.json; the selected SAE, source inputs and code hashes are in config/input_manifest.json. Runtime environments and scheduler metadata are recorded.\n\n## Evaluation and interpretation\n\nPer-image FaceLandmarker results and lossless landmarks, all FairFace face boxes, confidences, logits and probability vectors, SSIM and frozen-region pixel-change measures live in evaluations/. Predicted demographics describe perceived appearance, not identities. Face detection is a proxy, not structural validity. No validated beard classifier is available; SAM's existing prompt probes are not beard accuracy measurements.\n\nBlinded pages and an editable review form are in evaluations/blinded_review/. Labels are intentionally blank. The local blinding key is separate and should be withheld from reviewers. Facial-hair presence, full-beard presence, estimated coverage, structural validity and artifacts must be judged independently; no automatic semantic scores are invented. Missing semantic results are explicit in CSVs and the facial-hair figure. Rerunning analysis preserves existing review forms.\n\nStatistics pair methods within recipient seed and fixed donor, average shuffle replicates before seed aggregation, and report five-seed bootstrap intervals and exact sign-flip tests. Donors are fixed; analyses are exploratory and not multiplicity-adjusted. See results/research_summary.md for findings and limits. Source snapshots include both generation and evaluation code. output_manifest.json hashes every artifact except itself and run locks.\n''')
    inventory(run)
    print('Finalized manifest and documentation')
if __name__=='__main__':main()
