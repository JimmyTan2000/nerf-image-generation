import numpy as np

# Load each .npz file
data1 = np.load('generated_datasets/temp_dataset_chair_new1.pt.npz')
data2 = np.load('generated_datasets/temp_dataset_chair_new2.pt.npz')
data3 = np.load('generated_datasets/temp_dataset_chair_new3.pt.npz')

# Extract arrays
focal1, images1, poses1 = data1['focal'], data1['images'], data1['poses']
focal2, images2, poses2 = data2['focal'], data2['images'], data2['poses']
focal3, images3, poses3 = data3['focal'], data3['images'], data3['poses']

# Check focal lengths are the same (optional sanity check)
if not (np.allclose(focal1, focal2) and np.allclose(focal1, focal3)):
    raise ValueError("Focal lengths differ across files!")

# Concatenate the images and poses
combined_images = np.concatenate([images1, images2, images3], axis=0)
combined_poses = np.concatenate([poses1, poses2, poses3], axis=0)

# Save to new .npz file
np.savez('generated_datasets/combined/combined_data_chair.npz', focal=focal1, images=combined_images, poses=combined_poses)
