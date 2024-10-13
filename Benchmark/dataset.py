import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision import tv_tensors

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

import h5py



########################


class BraTS20_dataset(Dataset):
    def __init__(self, csv_file_path, memory=False): # transforms, 
        self.csv_file = pd.read_csv(csv_file_path)
        self.path = csv_file_path.split('/')
        # self.transforms = transforms

        self.memory = memory
        if memory:
            self._save_memory()

    def __len__(self,):
        return len(self.csv_file)
    
    def __getitem__(self, index):
         
        sample = self.csv_file.iloc[index]

        if self.memory :
            image = self.imgs[index] 
            mask = self.masks[index]#tv_tensors.Mask(self._mask_generator(self.masks[index]) )

        else :
            result = self._load(self._path_loader(sample['slice_path']))
            image = result[0]
            mask = self._mask_generator(result[1])#tv_tensors.Mask(self._mask_generator(result[1]))

        for i in range(image.shape[0]):    # Iterate over channels
                min_val = np.min(image[i])     # Find the min value in the channel
                image[i] = image[i] - min_val  # Shift values to ensure min is 0
                max_val = np.max(image[i]) + 1e-4     # Find max value to scale max to 1 now.
                image[i] = image[i] / max_val
        
        # image, mask = self.transforms(image, mask)
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask 
    
    def _save_memory(self):
        self.imgs = []
        self.masks = []
        for path in self.csv_file['slice_path']:
            sample = self._load(self._path_loader(path))
            self.imgs.append(sample[0])
            self.masks.append(self._mask_generator(sample[1]))


    def _load(self, path):
        with h5py.File(path, 'r') as file:
            image = file['image'][()].transpose(2, 0, 1)
            mask = file['mask'][()].transpose(2, 0, 1)

        return image, mask #tv_tensors.Image(image), mask 

    def _path_loader(self, x):
        return './' + self.path[1] + x[32:] #'/' + self.path[2] +
    
    def _mask_generator(self, x):
        combined_array = np.sum(x, axis=0)

        background = np.where(np.where(combined_array > 0, 1, 0) == 0, 1, 0).reshape(1, x.shape[1], x.shape[2])

        return np.concatenate((x, background), axis=0)
    


class BraTS20(object):
    def __init__(self, root, mode, mini=False, memory=True) :
        assert mode in ['train', 'valid', 'test'], 'mode should be train, test or valid'
        self.mini = mini  
        self.memory = memory

        # if mode == 'train' :
        #     self.path_dataset = root + "/train.csv"
        # elif mode == 'valid':
        #     self.path_dataset = root + "/valid.csv"
        # elif mode == 'test' :
        #     self.path_dataset = root + "/test.csv"

        self.path_dataset = root


    def __call__(self, batch_size) :
        dataset = BraTS20_dataset(self.path_dataset, memory=self.memory)   #self.transform,

        if self.mini == False :
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
       

        elif self.mini == True:
            dataset,_ = random_split(dataset,(1000, len(dataset)-1000))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader



if __name__=='__main__':
    csv_path = "./Dataset/BraTS20 Training Metadata.csv"

    # dataloader = UW_madison(root='./UW_madison_dataset', mode='train', mini=True, memory=True)(batch_size=32)

    # mask, img = next(iter(dataloader))

    # print(img[0].min(), img[0].max())
    # plt.imshow(img[0].permute(1,2,0) + mask[0].permute(1,2,0) * 0.5)
    # plt.show()

    dataset = BraTS20_dataset(csv_path)
    print(dataset.__len__())
    sample = dataset.__getitem__(1)
    print(sample[0].max(), sample[0].min())
    print(sample[0].shape, sample[1].shape)

    print('---------------------------')

    dataloader = BraTS20(csv_path, 'train', mini=False, memory=True)(batch_size=32)
    img, mask = next(iter(dataloader))
    print(mask.shape, img.shape)

    ## showing the image and mask

    fig, axes = plt.subplots(1, 4, figsize=(12, 5))  # 1 row, 2 columns

    
    axes[0].imshow(img[0][0])  
    axes[0].axis('off')      
    axes[0].set_title('T1-weighted') 

    
    axes[1].imshow(mask[0][0])  
    axes[1].axis('off')      
    axes[1].set_title('Necrotic') 

    axes[2].imshow(mask[0][1])  
    axes[2].axis('off')      
    axes[2].set_title('Edema') 

    axes[3].imshow(mask[0][2])  
    axes[3].axis('off')      
    axes[3].set_title('Tumour') 


    # Show the plot
    plt.tight_layout()  # Adjusts the layout to prevent overlap
    plt.show()



     