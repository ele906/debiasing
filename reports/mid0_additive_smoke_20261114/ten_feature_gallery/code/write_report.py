"""Document observed results without substituting an assumed null or race-causal conclusion."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--publish',type=Path);a=p.parse_args();run=a.run.resolve()
assert (run/'results/complete.json').exists(),'Wait for all existing-image metrics'
rows=list(csv.DictReader((run/'results/per_image_pixel_results.csv').open()));summary=list(csv.DictReader((run/'results/feature_strength_summary.csv').open()));cfg=json.loads((run/'config/config.json').read_text())
assert len(rows)==4200
summary={(int(r['feature']),r['strength'],r['activity']):r for r in summary}
exact=sum(r['pixels_identical']=='True' for r in rows)
allmae=np.array([float(r['pixel_mae']) for r in rows]);ssim=np.array([float(r['ssim']) for r in rows])
readme=['# Ten-feature additive interventions: old-photo gallery and research summary','',
 '**No new SDXL images were generated.** This analysis reuses all 4,200 completed additive edits: ten features × twelve prompt groups × five seeds × seven strengths. Original images and prior experiments are unchanged.','',
 '[Open the existing-photo gallery on GitHub](https://github.com/ele906/debiasing/tree/mid0-additive-smoke-20261114/reports/mid0_additive_smoke_20261114/ten_feature_gallery). The local gallery is in gallery/.','',
 '## Main conclusion','',
 '**The additive interventions are not literal no-ops. They change image pixels. Reliable semantic control has not been established, which is a different claim. These results do not show that race itself is the cause of the remaining appearance differences.**','',
 f'Exactly pixel-identical edited images: **{exact}/4,200**. Across the seven tested strengths, pixel MAE ranges from **{allmae.min():.4f} to {allmae.max():.4f}** on the 0–255 RGB scale; SSIM ranges from **{ssim.min():.4f} to {ssim.max():.4f}**. These quantities measure image differences, not beard presence, demographic identity or perceptual quality.','',
 '## Per-feature results','',
 'The following means use all twelve prompt groups and average within each of the five paired generation seeds. Matched-small and matched-moderate target applied block-output norms are feature-specific; they match groups within a feature, not dose across different features. Full 95% seed-bootstrap intervals and active/inactive summaries are in results/feature_strength_summary.csv.','',
 '| Feature | Initially inactive contexts /60 | Matched-small MAE | Matched-moderate MAE | Add+50 MAE | Exact pixel no-ops /420 |','|---|---:|---:|---:|---:|---:|']
latexrows=[]
for f in cfg['features']:
 subset=[r for r in rows if int(r['feature_id'])==f];contexts={r['prompt_id']:r['original_activation_zero']=='True' for r in subset};inactive=sum(contexts.values())
 small=summary[(f,'matched_small','all')];moderate=summary[(f,'matched_moderate','all')];strong=summary[(f,'fixed_50','all')]
 nzero=sum(r['pixels_identical']=='True' for r in subset)
 readme.append(f"| {f} | {inactive} | {float(small['mean_pixel_mae']):.3f} | {float(moderate['mean_pixel_mae']):.3f} | {float(strong['mean_pixel_mae']):.3f} | {nzero} |")
 latexrows.append(" & ".join([str(f),str(inactive)+"/60",format(float(small["mean_pixel_mae"]),".3f"),format(float(moderate["mean_pixel_mae"]),".3f"),format(float(strong["mean_pixel_mae"]),".3f"),str(nzero)+"/420"])+" "+chr(92)*2)
readme +=['','## How the interventions differ','',
 'Multiplication changes the selected coefficient by (alpha−1)z. It is a no-op when z is exactly zero. Addition changes it by beta at every spatial position and can introduce a previously inactive coordinate. The existing hook adds the corresponding decoder-direction change to the original block output, preserving the SAE reconstruction residual. Alpha=1 is therefore an exact no-op in this implementation; full SAE reconstruction is a separate control.','',
 'The seven additive levels are distribution-calibrated small and moderate offsets, two actual-norm-matched offsets, and fixed beta=10,20,50. Negative multipliers and the signed fixed-value sweep belong to earlier distinct experiments; this table does not silently combine them.','',
 '## What weak or inconsistent effects mean','',
 'The most defensible wording is: “The tested single-coordinate interventions produced measurable image changes, but the present measurements do not establish reliable, monotonic semantic control across prompt contexts.” Feature4228 should not be described as a proven beard feature.','',
 'This report does not measure beard coverage or other semantic attributes across all ten features. Nor does SSIM establish that identity, demographic appearance or image quality is preserved. No equivalence-to-zero threshold was specified, so small changes or lack of an obvious thumbnail difference cannot establish a formal null effect.','',
 '## Why this does not establish race as the cause','',
 'Prompt conditioning, other SAE coordinates, their interactions, spatial alignment, the rest of the diffusion representation, intervention timing, and evaluator sensitivity remain possible explanations. Race words in a prompt are conditioning variables; they are not an experimentally isolated biological cause.','',
 'A weak response to adding one coordinate does not isolate the mechanism that creates the original appearance. The coordinate may be insufficient, redundant, context-dependent, incorrectly interpreted, or edited in an unnatural spatial configuration. These alternatives are not distinguished by the present data. Neither “activations do nothing” nor “it must be race itself” follows.','',
 '## Scope and uncertainty','',
 'There are five paired generation seeds, not4,200 independent experimental units. Confidence intervals resample complete seed clusters while retaining all prompt groups. Active/inactive contexts are observational strata: their prompt-group composition differs, so their mean difference is not a randomized causal effect of activity. Dose comparisons across features must account for the recorded applied norms.','',
 'Checkpoint: layers/runs/mid0_sae_face_001/sae; SHA-256 cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6. Only unet.mid_block.attentions.0 and the same frozen single-step SDXL Turbo generator are represented.','',
 '## Files and reproduction','',
 'gallery/: derived contact sheets from saved images. results/gallery_sources.csv: source paths and hashes for every gallery cell. results/per_image_pixel_results.csv: all4,200 evaluated originals and paired baselines. results/feature_strength_summary.csv: seed-cluster intervals, active/inactive summaries and actual perturbation norms. report.tex and report.pdf: LaTeX research summary. code/: reproducible gallery and resumable image-metric scripts.','',
 'Run pixel_metrics.py --run THIS_DIRECTORY in the recorded existing analysis environment to resume missing image evaluations; no model is loaded. The image cache is append-only. All originals and activation archives remain in the earlier generation directories.','']
(run/'README.md').write_text('\n'.join(readme))
tex=r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,booktabs,graphicx,hyperref}
\hypersetup{hidelinks}
\title{Ten-Feature Additive SAE Interventions:\\ Measurable Image Changes and Limits of Semantic Interpretation}
\author{Exploratory research summary}
\date{November 16, 2026}
\begin{document}\maketitle
\begin{abstract}
We summarize existing single-coordinate interventions for ten features of a face-fine-tuned mid.0 sparse autoencoder (SAE). No new diffusion images are generated. The data comprise 4,200 edited images over twelve prompt groups, five paired seeds, and seven additive strengths. We distinguish nonzero image changes from reliable semantic control. Weak or inconsistent responses do not identify an intrinsic racial cause, nor prove that the tested coordinates have no effect.
\end{abstract}
\section{Question and experimental scope}
The original question is whether apparent demographic selectivity arises from sparse activation or from context-dependent intervention behavior. The SAE checkpoint is \path{layers/runs/mid0_sae_face_001/sae}, with SHA-256:
\begin{quote}\scriptsize\texttt{cbb3a0a2d0c43c822c7ba97a494de38ae56ea25401a8d31484ffa32e7cc329a6}\end{quote}
 The diffusion generator is the same frozen SDXL Turbo model; the layer is \path{unet.mid_block.attentions.0}. Sampling is single-step. No weights are trained or updated.
\section{Intervention definitions}
For selected feature $i$, decoder direction $w_i$, and original block output $h$, the residual-preserving intervention is
\[
 h'_{xy}=h_{xy}+\Delta z_i(x,y)w_i,\qquad
 \Delta z_i^{\mathrm{mult}}=(\alpha-1)z_i,\quad
 \Delta z_i^{\mathrm{add}}=\beta.
\]
Multiplication cannot change a coordinate that is zero. Addition can introduce that coordinate at all spatial positions. Neither intervention guarantees a monotonic semantic effect. Uniform addition can exceed the encoder's natural Top-$K$ support and does not reproduce a naturally localized activation pattern. The experiment records actual block-output perturbation norms after precision casting.
\section{Existing-image measurements}
Pixel MAE is the mean absolute RGB difference from the paired no-intervention image on the $0$--$255$ scale. SSIM is a structural-similarity proxy; neither is a validated beard, demographic, identity, or perceptual-quality measurement.
'''
tex+=f'Across all 4,200 edits, {exact} images are exactly pixel-identical to baseline. Pixel MAE ranges from {allmae.min():.4f} to {allmae.max():.4f}; SSIM ranges from {ssim.min():.4f} to {ssim.max():.4f}. Thus the edits are not generally literal no-ops.\n'
tex+=r'''\begin{table}[ht]\centering\small
\begin{tabular}{rrrrrr}\toprule
Feature & Inactive & Matched-small & Matched-moderate & Add $50$ & Exact no-ops\\
 & contexts & MAE & MAE & MAE & \\\midrule
'''
tex+='\n'.join(latexrows)+'\n'
tex+=r'''\bottomrule\end{tabular}
\caption{All prompt groups and five paired seeds. Matched perturbation targets differ across features; these columns are not a dose-controlled ranking of features. Inactive counts refer to sixty original prompt/seed contexts.}
\end{table}
\section{Interpretation and causal limits}
The appropriate conclusion is that the tested coordinates produce measurable image changes, while reliable, monotonic semantic control has not been established by these measurements. We do not summarize the result as ``additive features do nothing.'' Feature 4228 is a demographic-associated candidate, not a proven beard-control feature.

In a conditional generator $I=G(c,\epsilon;h)$, changing one coordinate while retaining prompt conditioning $c$ and the surrounding representation does not isolate the mechanism responsible for the original appearance. Prompt wording, other coordinates, spatial structure, feature interactions, intervention timing, and measurement sensitivity remain alternative explanations. A weak response to the tested edit therefore does not imply that ``race itself'' causes the outcome. Prompt race categories are conditioning text, not experimentally isolated biological variables.

Formal evidence of negligible semantic effects would require a defined semantic outcome, a justified equivalence margin, and adequate uncertainty assessment. Pixel similarity or an unchanged classifier label alone is insufficient. This analysis contains no validated all-feature beard or attribute-coverage measurement.
\section{Uncertainty and preservation}
The independent resampling units are five paired recipient seeds, not 4,200 independent images. Reported CSV confidence intervals resample whole seed clusters, retaining all twelve prompt groups. Active and inactive contexts are observational strata whose prompt composition differs. These comparisons are exploratory and cannot establish causal demographic specificity.

All original images and activation archives are retained. The gallery uses existing seed-30 images for all ten features and twelve prompt groups; selection is fixed rather than based on visually striking outcomes. Multiplicative and additive gallery columns are different parameterizations, not magnitude-matched comparisons. Source paths, hashes, per-image measurements and complete confidence intervals accompany this report.
\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{gallery/feature_4228_two_contexts.png}
\caption{Saved feature-4228 photos. Columns: unchanged, multiplication by7, addition of10,20,50. Rows: Black-man and East-Asian-man prompt contexts, seed30. These are viewing derivatives of existing originals, not new generations or evidence of a semantic null.}
\end{figure}
\end{document}
'''
(run/'report.tex').write_text(tex)
if a.publish:
 out=a.publish.resolve();shutil.copy2(run/'README.md',out/'research_summary.md');shutil.copy2(run/'report.tex',out/'report.tex')
 for name in ('per_image_pixel_results.csv','feature_strength_summary.csv'):shutil.copy2(run/'results'/name,out/'results'/name)
 text=(out/'README.md').read_text();text+='\n## Quantitative summary and LaTeX\n\n[Research summary](research_summary.md) · [LaTeX source](report.tex) · [PDF](report.pdf). These use all4,200 existing edits, not only the gallery examples.\n';(out/'README.md').write_text(text)
print('README and LaTeX written. Exact no-ops:',exact,'MAE range',allmae.min(),allmae.max())
