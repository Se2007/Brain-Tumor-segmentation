import torch
import segmentation_models_pytorch as smp


def UnetPlusPlus(encoder_name, in_channels=3, classes=3):
    return smp.UnetPlusPlus(encoder_name=encoder_name, encoder_depth=5, encoder_weights='imagenet', decoder_use_batchnorm=True, decoder_channels=(256, 128, 64, 32, 16), 
                            decoder_attention_type=None, in_channels=in_channels, classes=classes, activation=None, aux_params=None)


if __name__=='__main__':

    u = UnetPlusPlus(encoder_name='efficientnet-b1')

    print(u(torch.randn((1,3,224,224))).shape)