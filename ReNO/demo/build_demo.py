import json

data = json.load(open('/tmp/claude-365840/-n-fs-goose/4d75956c-9e7d-4f0e-ad0f-cef1c401c693/scratchpad/demo_data.json'))

html_template = r"""<title>ReNO — Reward-Based Noise Optimization</title>
<style>
:root {
  --bg: #17161c;
  --surface: #1f1e26;
  --surface-2: #29283349;
  --text: #ece8e1;
  --muted: #948fa0;
  --accent: #d9a441;
  --accent-soft: #d9a44122;
  --line: rgba(236,232,225,0.11);
  --display: Georgia, 'Iowan Old Style', 'Palatino Linotype', 'Times New Roman', serif;
  --body: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  --mono: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace;
}
:root[data-theme="light"] {
  --bg: #f8f6f1;
  --surface: #ffffff;
  --surface-2: #efeae0;
  --text: #221f28;
  --muted: #6b6577;
  --accent: #a9711a;
  --accent-soft: #a9711a1a;
  --line: rgba(34,31,40,0.12);
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    --bg: #f8f6f1;
    --surface: #ffffff;
    --surface-2: #efeae0;
    --text: #221f28;
    --muted: #6b6577;
    --accent: #a9711a;
    --accent-soft: #a9711a1a;
    --line: rgba(34,31,40,0.12);
  }
}
* { box-sizing: border-box; }
html, body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--body);
  margin: 0;
  padding: 0;
}
body {
  padding: 5rem 1.5rem 6rem;
}
.page {
  max-width: 42rem;
  margin: 0 auto;
}
.eyebrow {
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 0.9rem;
}
h1 {
  font-family: var(--display);
  font-weight: 400;
  font-size: 2.6rem;
  line-height: 1.1;
  margin: 0 0 1rem;
  text-wrap: balance;
}
h2 {
  font-family: var(--display);
  font-weight: 400;
  font-size: 1.5rem;
  line-height: 1.25;
  margin: 0 0 0.6rem;
  text-wrap: balance;
}
p {
  font-size: 1rem;
  line-height: 1.65;
  color: var(--text);
  max-width: 38rem;
  margin: 0 0 1rem;
}
p.lede {
  font-size: 1.1rem;
  color: var(--muted);
  line-height: 1.6;
}
.muted { color: var(--muted); }
.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 1.5rem 0 0;
}
.tag {
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.02em;
  border: 1px solid var(--line);
  color: var(--muted);
  padding: 0.3rem 0.6rem;
  border-radius: 3px;
}
section {
  margin-top: 5rem;
  padding-top: 3rem;
  border-top: 1px solid var(--line);
}
section:first-of-type { border-top: none; padding-top: 0; margin-top: 4rem; }
.section-num {
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: 0.08em;
  margin-bottom: 0.5rem;
}

/* Optimization strip */
.opt-frame {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 1.25rem;
  margin-top: 1.75rem;
}
.opt-image-wrap {
  aspect-ratio: 1 / 1;
  width: 100%;
  max-width: 22rem;
  margin: 0 auto;
  border-radius: 4px;
  overflow: hidden;
  background: var(--surface-2);
  position: relative;
}
.opt-image-wrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.opt-controls {
  max-width: 22rem;
  margin: 1rem auto 0;
}
.opt-slider-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
input[type="range"] {
  flex: 1;
  accent-color: var(--accent);
  height: 4px;
}
.opt-readout {
  font-family: var(--mono);
  font-size: 0.78rem;
  color: var(--accent);
  min-width: 5.5rem;
  text-align: right;
}
.opt-caption {
  font-family: var(--mono);
  font-size: 0.75rem;
  color: var(--muted);
  text-align: center;
  margin-top: 0.6rem;
}
.opt-prompt {
  text-align: center;
  font-family: var(--display);
  font-style: italic;
  color: var(--muted);
  font-size: 0.95rem;
  margin-top: 0.2rem;
}

/* Bias drift gallery */
.pair-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.25rem;
  margin-top: 1.75rem;
}
@media (max-width: 640px) {
  .pair-grid { grid-template-columns: 1fr; }
}
.pair-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.9rem;
}
.pair-images {
  display: flex;
  gap: 0.4rem;
}
.pair-images figure {
  margin: 0;
  flex: 1;
  min-width: 0;
}
.pair-images img {
  width: 100%;
  aspect-ratio: 1/1;
  object-fit: cover;
  border-radius: 3px;
  display: block;
}
.pair-images figcaption {
  font-family: var(--mono);
  font-size: 0.65rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
  text-align: center;
  margin-top: 0.35rem;
}
.pair-label {
  font-family: var(--display);
  font-style: italic;
  font-size: 0.92rem;
  color: var(--text);
  margin-top: 0.75rem;
  text-align: center;
}

/* Composition plots */
.plot-toggle {
  display: inline-flex;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.2rem;
  margin-top: 1.75rem;
  gap: 0.2rem;
}
.plot-toggle button {
  font-family: var(--mono);
  font-size: 0.75rem;
  letter-spacing: 0.03em;
  background: none;
  border: none;
  color: var(--muted);
  padding: 0.45rem 1rem;
  border-radius: 999px;
  cursor: pointer;
}
.plot-toggle button.active {
  background: var(--accent-soft);
  color: var(--accent);
}
.plot-wrap {
  margin-top: 1rem;
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
  background: var(--surface);
}
.plot-wrap img {
  width: 100%;
  display: block;
}
.plot-note {
  font-family: var(--mono);
  font-size: 0.75rem;
  color: var(--muted);
  margin-top: 0.6rem;
}

footer {
  margin-top: 5rem;
  padding-top: 2rem;
  border-top: 1px solid var(--line);
  font-family: var(--mono);
  font-size: 0.75rem;
  color: var(--muted);
  line-height: 1.7;
}
footer a { color: var(--muted); }
code.cite {
  display: block;
  white-space: pre-wrap;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 0.9rem;
  margin-top: 0.75rem;
  font-size: 0.7rem;
  color: var(--muted);
}
</style>

<div class="page">

  <div class="eyebrow">Inference-time optimization · Text-to-image</div>
  <h1>ReNO: steering a one-step diffusion model without retraining it</h1>
  <p class="lede">One-step models like SDXL-Turbo generate an image from a single forward pass — there's no iterative denoising loop to intervene in. ReNO instead treats the <em>initial noise</em> as the free variable: it backpropagates gradients from reward models straight through the frozen generator and climbs the noise toward images those rewards score higher.</p>
  <div class="tag-row">
    <span class="tag">ImageReward</span>
    <span class="tag">HPSv2</span>
    <span class="tag">PickScore</span>
    <span class="tag">CLIPScore</span>
    <span class="tag">50 iters · gradient ascent</span>
  </div>

  <section>
    <div class="section-num">01 — the mechanism</div>
    <h2>Watching the noise get optimized</h2>
    <p>Below is one real run: the prompt <span class="muted">"goose"</span> generated with SDXL-Turbo, gradient step 0 through the final selected frame. Drag the slider through the actual saved checkpoints — the model itself never changes, only the noise it starts from.</p>

    <div class="opt-frame">
      <div class="opt-image-wrap">
        <img id="opt-img" src="" alt="Optimization frame" />
      </div>
      <div class="opt-prompt">"a photo of a goose"</div>
      <div class="opt-controls">
        <div class="opt-slider-row">
          <input type="range" id="opt-slider" min="0" max="51" step="1" value="0" />
        </div>
        <div class="opt-caption" id="opt-caption">step 0 / init noise</div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-num">02 — the failure mode</div>
    <h2>Reward climbing has a direction — and it isn't neutral</h2>
    <p>Reward models are trained on human preference data, and that data has its own demographic skew. Optimize purely for "looks good / matches the prompt" on an occupation prompt like <em>nurse</em> or <em>CEO</em>, and the population of faces the model converges toward can shift — not because anyone asked for that, but because it's the direction the reward gradient points.</p>

    <div class="pair-grid" id="pair-grid"></div>
  </section>

  <section>
    <div class="section-num">03 — the experiment</div>
    <h2>Tracking demographic composition across optimization</h2>
    <p>To measure the drift directly, we generate 20 seeds per occupation prompt, classify the detected faces at every optimization stage with a FairFace model, and log the racial composition as the noise moves toward higher reward. <span class="muted">Baseline</span> uses the standard multi-reward objective; <span class="muted">fairness</span> adds a composition-aware term that regularizes the reward gradient against runaway demographic drift.</p>

    <div class="plot-toggle" id="plot-toggle">
      <button data-variant="baseline" class="active">baseline</button>
      <button data-variant="fairness">fairness-regularized</button>
    </div>
    <div class="plot-wrap">
      <img id="plot-img" src="" alt="Composition plot" />
    </div>
    <div class="plot-note" id="plot-note">a_photo_of_a_ceo · a_photo_of_a_nurse · a_photo_of_a_software_engineer — composition per optimization stage, 20 seeds each</div>
  </section>

  <footer>
    ReNO — Eyring, Karthik, Roth, Dosovitskiy, Akata (NeurIPS 2024) · arXiv:2406.04312
    <code class="cite">@article{eyring2024reno,
  title={ReNO: Enhancing One-step Text-to-Image Models through Reward-based Noise Optimization},
  author={Luca Eyring and Shyamgopal Karthik and Karsten Roth and Alexey Dosovitskiy and Zeynep Akata},
  journal={NeurIPS}, year={2024}}</code>
  </footer>
</div>

<script id="demo-data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('demo-data').textContent);

// --- optimization strip ---
const gooseFrames = ['init_image', ...Array.from({length: 50}, (_, i) => String(i)), 'best_image'];
const gooseLabels = ['step 0 / init noise', ...Array.from({length: 50}, (_, i) => `step ${i + 1}`), 'best / final noise'];
const optImg = document.getElementById('opt-img');
const optCaption = document.getElementById('opt-caption');
const optSlider = document.getElementById('opt-slider');
function renderOpt(i) {
  optImg.src = DATA.goose[gooseFrames[i]];
  optCaption.textContent = gooseLabels[i];
}
optSlider.addEventListener('input', e => renderOpt(+e.target.value));
renderOpt(0);

// --- bias drift pairs ---
const pairs = [
  { key: 'nurse', label: 'a photo of a nurse' },
  { key: 'ceo', label: 'a photo of a CEO' },
  { key: 'mechanic', label: 'a photo of a mechanic' },
];
const grid = document.getElementById('pair-grid');
pairs.forEach(p => {
  const card = document.createElement('div');
  card.className = 'pair-card';
  card.innerHTML = `
    <div class="pair-images">
      <figure><img src="${DATA[p.key].init}" alt="init"><figcaption>init</figcaption></figure>
      <figure><img src="${DATA[p.key].best}" alt="optimized"><figcaption>optimized</figcaption></figure>
    </div>
    <div class="pair-label">"${p.label}"</div>
  `;
  grid.appendChild(card);
});

// --- composition plot toggle ---
const plotImg = document.getElementById('plot-img');
const plotToggle = document.getElementById('plot-toggle');
function renderPlot(variant) {
  plotImg.src = DATA['plot_' + variant];
  [...plotToggle.children].forEach(b => b.classList.toggle('active', b.dataset.variant === variant));
}
plotToggle.addEventListener('click', e => {
  const btn = e.target.closest('button');
  if (!btn) return;
  renderPlot(btn.dataset.variant);
});
renderPlot('baseline');
</script>
"""

out = html_template.replace('__DATA__', json.dumps(data))
with open('/tmp/claude-365840/-n-fs-goose/4d75956c-9e7d-4f0e-ad0f-cef1c401c693/scratchpad/reno_demo.html', 'w') as f:
    f.write(out)
print('written', len(out))
