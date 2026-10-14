# Interactive image generation

Open either notebook from the repository root:

```bash
jupyter lab scripts/generate/generate_images.ipynb
jupyter lab scripts/generate/compare_models.ipynb
```

`generate_images.ipynb` lets you choose SDXL Turbo or SDXL Base, enter a prompt, and generate a seeded batch. `compare_models.ipynb` runs one prompt through both models. The shared `generation_utils.py` saves each image and its settings beneath `outputs/interactive_generation/`.

Run on a CUDA GPU. The notebook kernel needs `torch`, `diffusers`, `safetensors`, `ipywidgets`, and Jupyter. Model weights are downloaded from Hugging Face on first use, so accept the SDXL Base model terms and authenticate if Hugging Face requests it. Turbo uses one step at 512 × 512; Base defaults to 30 steps at 1024 × 1024. Base generation needs substantially more GPU memory.

The saved settings make runs reproducible at the prompt, model, seed, and sampler-parameter level. Different models do not produce paired images just because they share the same numeric seed.
