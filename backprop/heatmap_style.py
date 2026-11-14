"""Blocky jet overlays matching the original SDXL-Unbox notebooks."""
import numpy as np
from PIL import Image
from matplotlib import colormaps
from matplotlib.colors import ListedColormap


def make_heatmap_overlay(image, feature_map, alpha=.6):
    values = np.asarray(feature_map)
    minimum, maximum = float(values.min()), float(values.max())
    normalized = (values - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(values)
    jet = colormaps["jet"]
    colors = jet(np.arange(jet.N))
    colors[0, -1] = 0
    colors[1:, -1] = alpha
    rgba = ListedColormap(colors)(normalized)
    heat = Image.fromarray((rgba * 255).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
    return Image.alpha_composite(image.convert("RGBA"), heat)
