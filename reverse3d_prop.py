from torch import nn
from prop_model import CNNpropCNN_default
from unet import UnetGenerator, init_weights

class Reverse3dProp(nn.modules):
    def __init__(self) -> None:
        super().__init__()
        
        num_downs_slm = 8
        num_feats_slm_min = 32
        num_feats_slm_max = 512
        norm = nn.InstanceNorm2d
        self.reverse_cnn = UnetGenerator(input_nc= 2, output_nc=1,
                                    num_downs=num_downs_slm, nf0=num_feats_slm_min,
                                    max_channels=num_feats_slm_max, norm_layer=norm, outer_skip=True)
        init_weights(self.reverse_cnn, init_type='normal')
        
        self.CNNpropCNN = CNNpropCNN_default()
    
    def forward(self, input):
        slm_phase = self.reverse_cnn(input)
        reconstuct = self.CNNpropCNN(slm_phase)