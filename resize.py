import numpy as np
import torch
import torch.nn.functional as F

input_npz = "generated_datasets/temp_dataset_lego_new1.pt.npz"
output_npz = "generated_datasets/temp_dataset_lego_resized128.npz"
target_size = 128
batch_size = 100  # Adjust as needed

print(f"Loading: {input_npz}")
npz = np.load(input_npz)
images = npz["images"]  # [N, H, W, C]
print("Original shape:", images.shape, images.dtype)
print("Original image min/max:", images.min(), images.max())

N = images.shape[0]
resized_list = []

for start in range(0, N, batch_size):
    end = min(start + batch_size, N)
    print(f"Processing images {start} to {end-1}...")
    batch = images[start:end]  # [B, H, W, C]
    batch_torch = torch.from_numpy(batch).float().permute(0, 3, 1, 2)  # [B, C, H, W]

    # --- Value Range Fix ---
    # Scale according to input range
    min_val = batch_torch.min().item()
    max_val = batch_torch.max().item()
    if min_val >= 0.0 and max_val <= 1.0:
        print("Scaling images from [0, 1] to [0, 255]")
        batch_torch = batch_torch * 255.0
    elif min_val >= -1.0 and max_val <= 1.0:
        print("Scaling images from [-1, 1] to [0, 255]")
        batch_torch = (batch_torch + 1.0) * 127.5
    # Otherwise, assume already in [0, 255]

    print("Batch min/max BEFORE resize:", batch_torch.min().item(), batch_torch.max().item())

    with torch.no_grad():
        batch_resized = F.interpolate(
            batch_torch,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False
        )

    print("Batch min/max AFTER resize:", batch_resized.min().item(), batch_resized.max().item())

    batch_resized = batch_resized.permute(0, 2, 3, 1)  # [B, H, W, C]
    batch_resized = batch_resized.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    print("Batch min/max AFTER clamp/round:", batch_resized.min(), batch_resized.max())
    resized_list.append(batch_resized)
    # Clean up
    del batch, batch_torch, batch_resized

resized_np = np.concatenate(resized_list, axis=0)  # [N, H, W, C]
print("Resized shape:", resized_np.shape, resized_np.dtype)
print("Final min/max:", resized_np.min(), resized_np.max())
np.savez_compressed(output_npz, images=resized_np)
print(f"Saved resized images to: {output_npz}")
