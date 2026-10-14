# Start generation

Run commands from the repository root.

- ReNO single-prompt generation: `python scripts/generate_single_demo.py --prompt "parrot"`
- Generate SDXL Turbo images and cache SAE features for the existing backprop dataset: `python backprop/generate_images.py --count 100 --output backprop` (uses locally cached model weights and a CUDA GPU).

The second runner is a Turbo data-generation script, not the SDXL Base runner. The Base generation command and settings will be added with the first Base implementation.
