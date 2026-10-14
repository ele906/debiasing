# Activation and feature analysis

Run commands from the repository root unless a script says otherwise.

- Generate SDXL Turbo images and pooled SAE feature caches: `python backprop/generate_images.py --count 100 --output backprop`
- Capture spatial residuals for SAE box training: `python sae-training/capture_residuals.py`
- Analyze pooled SAE features: `python backprop/analyze_sae_features.py`
- Analyze flower feature localization and render overlays: `python flower_sae_intervention/analyze_feature_localization.py`

The current scripts and checkpoints are for SDXL Turbo. The SDXL Base experiment will need a separately captured activation set and Base-trained SAE.
