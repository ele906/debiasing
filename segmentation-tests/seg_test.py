import cv2
import numpy as np
import torch
import random
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 1. Load Image (MAKE SURE THIS PATH IS CORRECT)
image_path = "your_image.jpg" # <--- CHANGE THIS
image_bgr = cv2.imread("/n/fs/goose/data/datasets/biased/images/indian_man/seed015.png")
if image_bgr is None:
    raise ValueError(f"Could not load image at {image_path}. Check the path!")
    
print(f"Image loaded successfully. Shape: {image_bgr.shape}")
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 2. Load Model
sam = sam_model_registry["vit_b"](checkpoint="/scratch/el8403/checkpoints/sam_vit_b_01ec64.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")

# 3. Create the Mask Generator
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=1,
)

# 4. Generate the masks
print("Generating masks... (this takes a minute)")
masks = mask_generator.generate(image)
print(f"Found {len(masks)} masks.")

# 5. Visualize the map
def show_anns(anns, img):
    if len(anns) == 0:
        return img
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    overlay = img.copy()
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color = [random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)]
        overlay[m] = color
        
    blended = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    return blended

map_image = show_anns(masks, image)

# Save it
cv2.imwrite("segmentation_map.png", cv2.cvtColor(map_image, cv2.COLOR_RGB2BGR))
print("Saved segmentation_map.png!")