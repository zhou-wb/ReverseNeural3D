import torch
from torch import nn
from torch.nn import functional as F
from unet import UnetGenerator, init_weights
import utils
from torchsummary import summary
import prop_ideal
from algorithm import DPAC
from propagation_ASM import propagation_ASM


#####################
# U-Net Propagation #
#####################

class UNetProp(nn.Module):
    def __init__(self, img_size, input_nc, output_nc) -> None:
        super().__init__()
        
        num_downs = 8
        num_feats_min = 32
        num_feats_max = 512
        norm = nn.InstanceNorm2d
        self.unet = UnetGenerator(input_nc= input_nc, output_nc=output_nc,
                                    num_downs=num_downs, nf0=num_feats_min,
                                    max_channels=num_feats_max, norm_layer=norm, outer_skip=True)
        init_weights(self.unet, init_type='normal')
        
        self.img_size = img_size
        # the input size has to be the multiple of 2**num_downs for unet
        multiple = 2**num_downs
        self.reshape_size = ((img_size[0] + multiple - 1) // multiple * multiple, (img_size[1] + multiple - 1) // multiple * multiple)
    
    def forward(self, input):
        
        input = utils.pad_image(input, target_shape=self.reshape_size, pytorch=True, stacked_complex=False)
        input = utils.crop_image(input, target_shape=self.reshape_size, pytorch=True, stacked_complex=False)
        
        unet_output = self.unet(input)
        
        unet_output = utils.pad_image(unet_output, target_shape=self.img_size, pytorch=True, stacked_complex=False)
        unet_output = utils.crop_image(unet_output, target_shape=self.img_size, pytorch=True, stacked_complex=False)

        return unet_output



######################
# ResNet Propagation #
######################

class residual_block(nn.Module):
    def __init__(self,input_channel, output_channel, stride=1, downsample=None) -> None:
        super().__init__()
        self. downsample=downsample
        self.conv1=nn.Conv2d(input_channel,output_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(output_channel)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(output_channel,output_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(output_channel)
        self.relu2=nn.ReLU()
    
    
    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = out.clone() + identity
        return out

class ResNet_Prop(nn.Module):
    def __init__(self,input_channel=3,output_channel=2,block_num=30) -> None:
        super().__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.first_layer=nn.Sequential(
            nn.Conv2d(self.input_channel,24,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.layer1=self.make_layer(3, 24,block_num=1)
        self.layer2=self.make_layer(24,24,block_num=15)
        self.last_layer=nn.Sequential(
            nn.Conv2d(self.input_channel+24,self.output_channel,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU()
        )
    
    def forward(self,x):
        identity = x
        x = self.first_layer(x)
        # out=self.layer1(x)
        out=self.layer2(x)
        out = torch.cat((identity,out),dim=1) # concat channel
        out=self.last_layer(out)
        return out
    
    def make_layer(self, input_channel, output_channel, block_num=30,stride=1):
        layers=[]
        layers.append(residual_block(input_channel,output_channel))
        for _ in (1,block_num):
            layers.append(residual_block(output_channel,output_channel))
        return nn. Sequential(*layers)




####################################################
# Different Implementations of Inverse Propagation #
####################################################

class InversePropagation(nn.Module):
    def __init__(self, inverse_network_config, **kwargs):
        super().__init__()
        self.config = inverse_network_config
        if self.config == 'cnn_only':
            self.inverse_cnn = ResNet_Prop(input_channel=len(kwargs['prop_dists_from_wrp']), output_channel=1)
            # self.inverse_cnn = UNetProp(img_size=kwargs['image_res'], input_nc=len(kwargs['prop_dists_from_wrp']), output_nc=1)
            self.forward = self.inverse_CNN_only
        elif self.config == 'cnn_asm_dpac':
            self.target_cnn = ResNet_Prop(input_channel=len(kwargs['prop_dists_from_wrp']), output_channel=2)
            self.asm_dpac = DPAC(kwargs['prop_dist'], kwargs['wavelength'], kwargs['feature_size'], prop_model='ASM', 
                                 propagator=propagation_ASM, device=kwargs['device'])
            self.forward = self.inverse_CNN_ASM_DPAC
        elif self.config == 'cnn_asm_cnn':
            self.target_cnn = ResNet_Prop(input_channel=len(kwargs['prop_dists_from_wrp']), output_channel=2)
            self.inverse_asm = prop_ideal.SerialProp(-kwargs['prop_dist'], kwargs['wavelength'], kwargs['feature_size'],
                                        'ASM', kwargs['F_aperture'], None,
                                        dim=1)
            self.slm_cnn = UNetProp(img_size=kwargs['image_res'], input_nc=2, output_nc=1)
            self.forward = self.inverse_CNN_ASM_CNN
        else:
            raise ValueError(f'{inverse_network_config} not implemented!')
        
    def inverse_CNN_only(self, masked_imgs):
        ########## phase generated by CNN only ###################        
        slm_phase = self.inverse_cnn(masked_imgs)
        ##########################################################
        return slm_phase

    def inverse_CNN_ASM_DPAC(self, masked_imgs):
        ########## phase generated by CNN+ASM+DPAC ###############
        mid_amp_phase = self.target_cnn(masked_imgs)
        mid_amp = mid_amp_phase[:,0:1,:,:]
        mid_phase = mid_amp_phase[:,1:2,:,:]
        _, slm_phase = self.asm_dpac(mid_amp, mid_phase)
        ##########################################################
        return slm_phase

    def inverse_CNN_ASM_CNN(self, masked_imgs):
        ########## phase generated by CNN+ASM+CNN ###############
        mid_amp_phase = self.target_cnn(masked_imgs)
        mid_amp = mid_amp_phase[:,0:1,:,:]
        mid_phase = mid_amp_phase[:,1:2,:,:]
        mid_field = torch.complex(mid_amp * torch.cos(mid_phase), mid_amp * torch.sin(mid_phase))
        slm_field = self.inverse_asm(mid_field)
        slm_phase = self.slm_cnn(torch.cat([slm_field.abs(), slm_field.angle()], dim=1))
        ##########################################################
        return slm_phase
    
    def train(self):
        if self.config == 'cnn_only':
            self.inverse_cnn.train()
        elif self.config == 'cnn_asm_dpac':
            self.target_cnn.train()
        elif self.config == 'cnn_asm_cnn':
            self.target_cnn.train()
            self.slm_cnn.train()

    def eval(self):
        if self.config == 'cnn_only':
            self.inverse_cnn.eval()
        elif self.config == 'cnn_asm_dpac':
            self.target_cnn.eval()
        elif self.config == 'cnn_asm_cnn':
            self.target_cnn.eval()
            self.slm_cnn.eval()

    def save(self, path):
        if self.config == 'cnn_only':
            torch.save(self.inverse_cnn, path)
        elif self.config == 'cnn_asm_dpac':
            torch.save(self.target_cnn, path)
        elif self.config == 'cnn_asm_cnn':
            torch.save(self.target_cnn, path)
            torch.save(self.slm_cnn, path)

    def configoptimizer(self, props, lr):
        if self.config == 'cnn_only':
            return torch.optim.Adam(self.inverse_cnn.parameters(), lr=lr)
        elif self.config == 'cnn_asm_dpac':
            return torch.optim.Adam(self.target_cnn.parameters(), lr=lr)
        elif self.config == 'cnn_asm_cnn':
            return torch.optim.Adam(list(self.target_cnn.parameters()) + list(self.slm_cnn.parameters()), lr=lr)



if __name__ == '__main__':
    reverse_prop = UNetProp()
    reverse_prop = reverse_prop.cuda()
    summary(reverse_prop, (8, 1080, 1920))