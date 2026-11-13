import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# 1. Load Image
image_bgr = cv2.imread("/n/fs/goose/data/datasets/biased/images/indian_man/seed015.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 2. Load SAM
sam = sam_model_registry["vit_b"](checkpoint="/scratch/el8403/checkpoints/sam_vit_b_01ec64.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# 3. The Guessed Coordinates
points = np.array([
    [256, 400],  # 1. Center of the beard (Include)
    [220, 420],  # 2. Left side of beard (Include)
    [290, 420],  # 3. Right side of beard (Include)
    [190, 270],  # 4. Left cheek (Exclude - wall)
    [320, 270],  # 5. Right cheek (Exclude - wall)
    [256, 320],  # 6. Mustache/Nose area (Exclude - wall)
])

# 1 = include, 0 = exclude
labels = np.array([1, 1, 1, 0, 0, 0]) 

# 4. Predict
masks, scores, _ = predictor.predict(
    point_coords=points,
    point_labels=labels,
    multimask_output=False, 
)

mask = masks[0]

# 5. Save the Overlay so you can see the result
overlay_bgr = image_bgr.copy()
overlay_bgr[mask] = [0, 0, 255] # Paint it red
blended = cv2.addWeighted(overlay_bgr, 0.5, image_bgr, 0.5, 0)
cv2.imwrite("beard_overlay_guessed.png", blended)

# 6. Save the Cutout
rgba = np.dstack([image_rgb, mask.astype(np.uint8) * 255])
cv2.imwrite("beard_cutout_guessed.png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

print("Saved beard_overlay_guessed.png and beard_cutout_guessed.png")