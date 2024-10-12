import torch
from torch import nn
from torch import optim
from torchmetrics import Dice
import segmentation_models_pytorch as smp
from loss.loss import Criterion

from benchmark import dataset
from utils import train_one_epoch
from methods.unet import UNet, pre_train_unet
from methods.unetplusplus import UnetPlusPlus
from methods.deeplab import DeepLab
from methods.transunet import TransUNet
# from methods.vit import ViT

from prettytable import PrettyTable
from colorama import Fore, Style, init

## Function for load the model for change and find the hyperparameter during the training

def load(model, device='cpu', reset = False, load_path = None):
    model = model

    if reset == False : 
        if load_path is None :
            print('give path for load model')
        if load_path is not None:
            if device == 'cpu':
                sate = torch.load(load_path,map_location=torch.device('cpu'))
            else :
                sate = torch.load(load_path)
            
            model.load_state_dict(sate['state_dict'])
    return model

####  Arguments

device = 'cuda'
num_epochs = 5
reset = True

train_loader = dataset.UW_madison(root='./benchmark/UW_madison_dataset', mode='train', mini=True, memory=False)(batch_size=20)

load_path = './saved_model/'+'unet-with-dropout'+ ".pth"

#######################
#   Hyperparameters   #
#######################

learning_rates = [0.8,  1]
weight_decays = [1e-2, 1e-4, 1e-6]

## preprocessing for makeing the table and finding the minimums

loss_list = []

best_lr = None
best_wd = None
best_loss = float('inf')  
min_num = float('inf')
second_min = float('inf')

table = PrettyTable()
table.field_names = ["LR \ WD"] + [f"WD {i}" for i in weight_decays]

## Loss function and Metric

metric = Dice().to(device)

# loss_fn = smp.losses.DiceLoss(mode='multilabel')
loss_fn = Criterion()


for lr in learning_rates:
    for wd in weight_decays:
    
        print(f'\nLR={lr}, WD={wd}')

        ## Model and Optimizer
        
        # model = UNet(n_channels=3, n_classes=3, bilinear=False).to(device)

        # model = pre_train_unet(encoder_name='timm-regnety_008').to(device)

        # model = UnetPlusPlus(encoder_name='efficientnet-b3').to(device)

        # model = DeepLab(encoder_name='efficientnet-b1').to(device)

        model = TransUNet(img_dim=224, in_channels=3, class_num=3).to(device)

        ### Calculate the amount of parameters
        print(sum(p.numel() for p in model.parameters()))
        
        model = load(model, device=device, reset = reset, load_path = load_path)
        
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=False)


        for epoch in range(1, num_epochs+1):
            model, loss, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch, device=device)

     
        loss_list.append(float(f'{loss:.4f}'))

## Add the color to the first and second minimun loss of the table

sorted_list = sorted(loss_list)
first_min = sorted_list[0]
second_min = sorted_list[1]

first_min_idx = loss_list.index(first_min)
second_min_idx = loss_list.index(second_min)

loss_list[first_min_idx] = f"{Fore.GREEN}{first_min}{Fore.WHITE}"
loss_list[second_min_idx] = f"{Fore.YELLOW}{second_min}{Fore.WHITE}"
loss_list = list(map(str, loss_list))

## Making the table

o = 0

for i in learning_rates:
    row = [f"LR {i}"]

    losses = loss_list[o:len(weight_decays)+o]
    o += len(weight_decays)

    row += losses
    table.add_row(row)


print(table)