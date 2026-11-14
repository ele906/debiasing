import cv2
import numpy as np
import torch
import os
from segment_anything import sam_model_registry, SamPredictor

# Load SAM
print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint="/scratch/el8403/checkpoints/sam_vit_b_01ec64.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
predictor = SamPredictor(sam)

# Hardcoded prompts for the beard based on earlier tests
# [x, y] coords. 1 = include, 0 = exclude
points = np.array([
    [256, 400], [220, 420], [290, 420],  # beard
    [190, 270], [320, 270], [256, 320]   # cheeks/nose walls
])
labels = np.array([1, 1, 1, 0, 0, 0]) 

out_dir = "beard_cutouts"
os.makedirs(out_dir, exist_ok=True)

def process_image(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    
    masks, _, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False
    )
    mask = masks[0]
    
    # save RGBA
    out_name = os.path.basename(img_path).replace(".png", "_cutout.png")
    rgba = np.dstack([img_rgb, mask.astype(np.uint8) * 255])
    cv2.imwrite(os.path.join(out_dir, out_name), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    
    return mask, img_bgr

print("Processing seeds 000-099...")
previews = []

for i in range(100):
    seed = f"{i:03d}"
    path = f"/n/fs/goose/data/datasets/biased/images/indian_man/seed{seed}.png"
    
    if not os.path.exists(path):
        continue
        
    print(f"  seed{seed}...")
    res = process_image(path)
    
    # save the first 9 for the grid
    if res and len(previews) < 9:
        mask, img = res
        img[mask] = [0, 0, 255] # red overlay
        blended = cv2.addWeighted(img, 0.5, cv2.imread(path), 0.5, 0) # blend with clean original
        cv2.putText(blended, f"seed{seed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        previews.append(blended)

# Make the grid
if previews:
    print("Building grid...")
    # resize to 256x256
    small = [cv2.resize(p, (256, 256)) for p in previews]
        
    grid = np.vstack([
        np.hstack(small[0:3]),
        np.hstack(small[3:6]),
        np.hstack(small[6:9])
    ])
    cv2.imwrite("preview_grid.png", grid)
    print("Done")