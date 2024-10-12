import segmentation_models_pytorch as smp
from .unet_parts import *    


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.factor = 2 if bilinear else 1

        self.encoders = self._encoder_layer()

        self.decoders = self._decoder_layer()

    def _encoder_layer(self, num_layer=4):
        layers = []
        channel = 64

        layers.append(DoubleConv(self.n_channels, channel))

        for i in range(num_layer):
            if i+1 != num_layer :
                layers.append(Down(channel * 2**i, channel * 2**(i+1)))
            else:
                layers.append(Down(channel * 2**i, channel * 2**(i+1)))

        return nn.ModuleList(layers)
    
    def _decoder_layer(self, num_layer=4):
        layers = []
        channel = 64

        for i in range(num_layer,0,-1):
            layers.append(Up(channel * 2**i, channel * 2**(i-1) // self.factor, self.bilinear))

        layers.append(OutConv(64, self.n_classes))

        return nn.ModuleList(layers)

    def forward(self, x):
        intermediates = []
        for encoder in self.encoders:
            x = encoder(x)
            intermediates.append(x)

        for i, decoder in enumerate(self.decoders):
            if i+1 != len(self.decoders):
                x = decoder(x, intermediates[-i - 2])
            else:
                seg = decoder(x)

        
        return seg


def pre_train_unet(in_channels=3, classes=3, encoder_name='efficientnet-b1'):
    return smp.Unet(encoder_name=encoder_name, encoder_weights='imagenet',
                    in_channels=in_channels, classes=classes)



    
if __name__=='__main__':

    u = UNet(n_channels=3, n_classes=3, bilinear=False)

    print(u(torch.randn((1,3,224,224))).shape)
