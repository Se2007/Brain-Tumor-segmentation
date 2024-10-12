import torch
import segmentation_models_pytorch as smp


def DeepLab(encoder_name, in_channels=3, classes=3):
    return smp.DeepLabV3(encoder_name=encoder_name, encoder_depth=5, encoder_weights='imagenet', decoder_channels=265, in_channels=in_channels, classes=classes, activation=None, upsampling=8, aux_params=None)


if __name__=='__main__':

    d = DeepLab(encoder_name='efficientnet-b1')

    # print(d(torch.randn((10,3,512,512))).shape)
    print(d(torch.randn((2,3,224,224))).shape)
