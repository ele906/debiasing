import cv2
import numpy as np
import torch
import os
from segment_anything import sam_model_registry, SamPredictor

# --- SETUP ---
print("Loading SAM Model...")
sam = sam_model_registry["vit_b"](checkpoint="/scratch/el8403/checkpoints/sam_vit_b_01ec64.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)
os.makedirs("beard_cutouts", exist_ok=True) # output folder

# Fixed coordinates
points = np.array([
    [256, 400],  # Center
    [220, 420],  # Left
    [290, 420],  # Right
    [190, 270],  # Cheek L
    [320, 270],  # Cheek R
    [256, 320],  # Nose
])
labels = np.array([1, 1, 1, 0, 0, 0]) 

# --- Process Image ---
def process_image(image_path, output_dir):
    # Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Could not read {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Set image for SAM
    predictor.set_image(image_rgb)
    
    # Predict
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False, 
    )
    mask = masks[0]
    
    # Save Cutout
    filename = os.path.basename(image_path).replace(".png", "_cutout.png")
    out_path = os.path.join(output_dir, filename)
    
    rgba = np.dstack([image_rgb, mask.astype(np.uint8) * 255])
    cv2.imwrite(out_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    
    # Return the mask and original image for the preview grid
    return mask, image_bgr

# --- PROCESSING LOOP ---
print("Starting loop from 000 to 099...")
preview_images = []

for i in range(100): # 0 to 99
    # Format the number to be 3 digits (e.g., 0 -> 000, 15 -> 015)
    seed_str = f"{i:03d}"
    img_path = f"/n/fs/goose/data/datasets/biased/images/indian_man/seed{seed_str}.png"
    
    if not os.path.exists(img_path):
        continue # Skip if the file doesn't exist
        
    print(f"Processing seed{seed_str}...")
    
    result = process_image(img_path, "beard_cutouts")
    
    # keep 9 results for a preview grid
    if result is not None and len(preview_images) < 9:
        mask, img = result
        # Make a small red overlay for the preview
        overlay = img.copy()
        overlay[mask] = [0, 0, 255]
        blended = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        # Add text of the seed number
        cv2.putText(blended, f"seed{seed_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        preview_images.append(blended)

# --- VERIFICATION GRID ---
if len(preview_images) > 0:
    print("Building preview grid...")
    # Resize images to 256x256
    resized = [cv2.resize(img, (256, 256)) for img in preview_images]
    
    # If we have less than 9 imgs, pad with black img (fallback)
    while len(resized) < 9:
        resized.append(np.zeros((256, 256, 3), dtype=np.uint8))
        
    # Create 3x3 grid
    row1 = np.hstack(resized[0:3])
    row2 = np.hstack(resized[3:6])
    row3 = np.hstack(resized[6:9])
    grid = np.vstack((row1, row2, row3))
    
    cv2.imwrite("preview_grid.png", grid)
    print("Saved 100 cutouts to 'beard_cutouts/' and a preview to 'preview_grid.png'")