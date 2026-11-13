import cv2
import numpy as np
import torch
import random
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 1. Load Image
image_bgr = cv2.imread("/n/fs/goose/data/datasets/biased/images/indian_man/seed015.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 2. Create the Contrast-Boosted Image (CLAHE)
lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
boosted_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
boosted_rgb = cv2.cvtColor(boosted_bgr, cv2.COLOR_BGR2RGB)

# Save the boosted image just so you can see what it looks like
cv2.imwrite("boosted_image.png", boosted_bgr)

# 3. Load SAM
sam = sam_model_registry["vit_b"](checkpoint="/scratch/el8403/checkpoints/sam_vit_b_01ec64.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")

# 4. Create the Mask Generator
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=1,
)

# 5. Generate the masks on the BOOSTED image
print("Generating masks on boosted image... (this takes a minute)")
masks = mask_generator.generate(boosted_rgb)
print(f"Found {len(masks)} masks.")

# 6. Visualize the map
def show_anns(anns, img):
    if len(anns) == 0:
        return img
    
    # Sort masks by area (largest first)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    overlay = img.copy()
    
    for ann in sorted_anns:
        m = ann['segmentation']
        # Generate a random bright color
        color = [random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)]
        # Paint the mask area with the color
        overlay[m] = color
        
    # Blend the boosted image (40%) with the colored overlay (60%)
    blended = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    return blended

# Create the final map
map_image = show_anns(masks, boosted_rgb)

# Save it (map_image is in RGB, so we convert to BGR for cv2.imwrite)
cv2.imwrite("boosted_segmentation_map.png", cv2.cvtColor(map_image, cv2.COLOR_RGB2BGR))
print("Saved boosted_segmentation_map.png!")