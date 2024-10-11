import torch 
import cv2
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import h5py

csv_path = "./Dataset/BraTS20 Training Metadata.csv"

train_df = pd.read_csv(csv_path)

paths = train_df[train_df['volume'] == 20]['slice_path'].values.tolist()
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

for image, mask in zip(images, masks) :
    mask_image = overlay_masks_on_image(image, mask)
    image = image[0, :, :]
    # mask = mask_trasform(rle2mask([sample.iloc[0]['height'], sample.iloc[0]['weidth']], [sample.iloc[0]['large_bowel'], sample.iloc[0]['small_bowel'], sample.iloc[0]['stomach']]))

    # out = image.permute(1,2,0).numpy() + mask.permute(1,2,0).numpy() * 0.55


       
    # frame_rgb = (out * 255).astype(np.uint8)
    # image_lst.append(frame_rgb)
    array_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored_image = cv2.applyColorMap(array_normalized, cv2.COLORMAP_MAGMA)

    cv2.imshow('R', colored_image)
    cv2.imshow('Red : large_bowel -- Green : small_bowel -- Blue : stomach', mask_image)
    cv2.waitKey(100)


## for making the Gif, have to install imageio

# imageio.mimsave('./v.gif', image_lst)