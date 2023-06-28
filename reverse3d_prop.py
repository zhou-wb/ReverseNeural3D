from torch import nn
# from prop_model import CNNpropCNN_default
from unet import UnetGenerator, init_weights
import utils
from torchsummary import summary

class Reverse3dProp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        num_downs_slm = 8
        num_feats_slm_min = 32
        num_feats_slm_max = 512
        norm = nn.InstanceNorm2d
        self.reverse_cnn = UnetGenerator(input_nc= 8, output_nc=2,
                                    num_downs=num_downs_slm, nf0=num_feats_slm_min,
                                    max_channels=num_feats_slm_max, norm_layer=norm, outer_skip=True)
        init_weights(self.reverse_cnn, init_type='normal')
        
        # self.CNNpropCNN = CNNpropCNN_default()
        
        pass
    
    def forward(self, input):
        
        input = utils.pad_image(input, target_shape=(1280, 2048), pytorch=True, stacked_complex=False)
        input = utils.crop_image(input, target_shape=(1280, 2048), pytorch=True, stacked_complex=False)
        
        slm_phase = self.reverse_cnn(input)      # input.shape = torch.Size([1, 8, 1080, 1920])
        
        slm_phase = utils.pad_image(slm_phase, target_shape=(1080,1920), pytorch=True, stacked_complex=False)
        slm_phase = utils.crop_image(slm_phase, target_shape=(1080,1920), pytorch=True, stacked_complex=False)

        
        # reconstuct = self.CNNpropCNN(slm_phase)  
        
        return slm_phase


if __name__ == '__main__':
    reverse_prop = Reverse3dProp()
    reverse_prop = reverse_prop.cuda()
    summary(reverse_prop, (8, 1080, 1920))