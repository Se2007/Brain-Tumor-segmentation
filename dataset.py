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

#############################
# Define seg-areas
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3
}

# Select Slices and Image Size
VOLUME_SLICES = 100
VOLUME_START_AT = 22 # first slice of volume that we will include
IMG_SIZE=128

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 4))


        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii');
            flair = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_t1ce.nii');
            t1ce = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_seg.nii');
            seg = nib.load(data_path).get_fdata()

            for j in range(VOLUME_SLICES):
                 X[j +VOLUME_SLICES*c,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
                 X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(t1ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));

                 y[j +VOLUME_SLICES*c] = seg[:,:,j+VOLUME_START_AT];

        # Generate masks
        y[y==4] = 3;
        mask = tf.one_hot(y, 4);
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE));
        return X/np.max(X), Y

training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)



########################


class BraTS20_dataset(Dataset):
    def __init__(self, csv_file_path, transforms, memory=False):
        self.csv_file = pd.read_csv(csv_file_path)
        self.transforms = transforms

        self.memory = memory
        if memory:
            self._save_memory()

    def __len__(self,):
        return len(self.csv_file)
    
    def __getitem__(self, index):
         
        sample = self.csv_file.iloc[index]

        if self.memory :
            image = self.imgs[index] 
            mask = self.masks[index] 

        else :
            pack = self._load(self._path_loader(sample['slice_path']))
            image = pack[0]
            mask = pack[1]

        
        
        # image, mask = self.transforms(image, mask)

        return image, mask 
    
    def _save_memory(self):
        self.imgs = []
        self.masks = []
        for path in self.csv_file['slice_path']:
            sample = self._load(self._path_loader(path))
            self.imgs.append(sample[0])
            self.masks.append(sample[1])


    def _load(self, path):
        with h5py.File(path, 'r') as file:
            image = file['image'][()].transpose(2, 0, 1)
            mask = file['mask'][()].transpose(2, 0, 1)

        return image, mask 

    def _path_loader(self, x):
        return './Dataset' + x[32:]
    


class UW_madison(object):
    def __init__(self, root, mode, mini=False, memory=True) :
        assert mode in ['train', 'valid', 'test'], 'mode should be train, test or valid'
        self.mini = mini  
        self.memory = memory
        
        self.transform = v2.Compose([v2.Resize(size=(234, ), antialias=True),
                           v2.RandomCrop(size=(224, 224)),
                           v2.RandomPhotometricDistort(p=.5),
                           v2.RandomHorizontalFlip(p=.5),
                           v2.ElasticTransform(alpha=50),
                           v2.ToTensor(),
                           v2.Lambda(lambda x : (x - x.min()) / (x.max() - x.min())),
                           v2.Normalize(mean=(0.5, ), std=(0.5, )),
                           v2.Lambda(lambda x : x.repeat(3, 1, 1))
                           ])
            

        if mode == 'train' :
            self.path_dataset = root + "/train.csv"
        elif mode == 'valid':
            self.path_dataset = root + "/valid.csv"
        elif mode == 'test' :
            self.path_dataset = root + "/test.csv"


    def __call__(self, batch_size) :
        dataset = UW_madison_dataset(self.path_dataset, self.transform, memory=self.memory)

        if self.mini == False :
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

        elif self.mini == True:
            dataset,_ = random_split(dataset,(1000, len(dataset)-1000))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader



if __name__=='__main__':


    dataloader = UW_madison(root='./UW_madison_dataset', mode='train', mini=True, memory=True)(batch_size=32)

    mask, img = next(iter(dataloader))

    print(img[0].min(), img[0].max())
    plt.imshow(img[0].permute(1,2,0) + mask[0].permute(1,2,0) * 0.5)
    plt.show()


     