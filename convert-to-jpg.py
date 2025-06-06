import os
import numpy as np
from PIL import Image

input_npz = "generated_datasets/temp_dataset_lego_resized128.npz"
output_dir = "generated_datasets/lego_jpgs128"
os.makedirs(output_dir, exist_ok=True)

# Load the resized images
print(f"Loading images from: {input_npz}")
npz = np.load(input_npz)
images = npz["images"]  # [N, H, W, C]
print("Loaded images shape:", images.shape, images.dtype)

# Save images as JPG
for idx, img in enumerate(images):
    # Convert to PIL Image
    pil_img = Image.fromarray(img)
    out_path = os.path.join(output_dir, f"{idx:05d}.jpg")
    pil_img.save(out_path, format="JPEG", quality=95)
    if idx % 100 == 0:
        print(f"Saved {idx + 1} / {images.shape[0]} images")

print(f"Done! All images saved to: {output_dir}")
