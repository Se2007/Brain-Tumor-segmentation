import torch 
import cv2
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import h5py
import imageio

csv_path = "./Dataset/BraTS20 Training Metadata.csv"

train_df = pd.read_csv(csv_path)

idx = int(input('MRI scans of which patient : '))

paths = train_df[train_df['volume'] == idx]['slice_path'].values.tolist()
images = []
masks = []
for idx, path in enumerate(paths):
    path = './Dataset' + path[32:]

    # sample_file_path = os.path.join(root, h5_files[20070])
    
    with h5py.File(path, 'r') as file:
        # print(file['image'][()].transpose(2, 0, 1).shape)
        images.append(file['image'][()].transpose(2, 0, 1))
        masks.append(file['mask'][()].transpose(2, 0, 1))

def overlay_masks_on_image(image, mask):
    t1_image = image[0, :, :]  # Use the first channel of the image
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())

    rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)
    color_mask = np.stack([mask[0, :, :], mask[1, :, :], mask[2, :, :]], axis=-1)
    rgb_image = np.where(color_mask, color_mask, rgb_image)
    
    return rgb_image

image_lst = []
for image, mask in zip(images, masks) :
    mask_image = overlay_masks_on_image(image, mask)
    image = image[0, :, :]

    array_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored_image = cv2.applyColorMap(array_normalized, cv2.COLORMAP_MAGMA)

    scale_factor = 3
    new_size = (colored_image.shape[1] * scale_factor, colored_image.shape[0] * scale_factor)
    
    colored_image_resized = cv2.resize(colored_image, new_size, interpolation=cv2.INTER_CUBIC)
    mask_image_resized = cv2.resize(mask_image, new_size, interpolation=cv2.INTER_CUBIC)

    frame_rgb = cv2.cvtColor((mask_image_resized * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    image_lst.append(frame_rgb)
    # if mask_image_resized.dtype != np.uint8:
    #     mask_image_resized = mask_image_resized.astype(np.uint8)

    # frame_rgb = cv2.cvtColor(mask_image_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # image_lst.append(frame_rgb)

    title_colored = 'Brain MRI Scans'
    title_mask = 'Red : Necrotic - Green : Edema - Blue : Tumour'

    cv2.putText(colored_image_resized, title_colored, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(mask_image_resized, title_mask, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('Brain MRI Scans', colored_image_resized)
    cv2.imshow('Red : Necrotic -- Green : Edema -- Blue : Tumour', mask_image_resized)
    cv2.waitKey(100)


## for making the Gif, have to install imageio

# imageio.mimsave('./11v.gif', image_lst)