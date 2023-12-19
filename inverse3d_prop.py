import torch
from torch import nn
import numpy as np
from unet import UnetGenerator, init_weights
import utils
from pytorch_prototyping.pytorch_prototyping import Unet

import prop_ideal
from algorithm import DPAC
from propagation_ASM import propagation_ASM
from model import Uformer

#####################
# U-Net Propagation #
#####################

class UNetProp(nn.Module):
    def __init__(self, img_size, input_nc, output_nc, num_downs=8) -> None:
        super().__init__()
        
        num_downs = num_downs
        num_feats_min = 32
        num_feats_max = 512
        norm = nn.InstanceNorm2d
        # self.unet = UnetGenerator(input_nc= input_nc, output_nc=output_nc,
        #                             num_downs=num_downs, nf0=num_feats_min,
        #                             max_channels=num_feats_max, norm_layer=norm, outer_skip=True)
        self.unet = Unet(input_nc, output_nc, 
                         nf0=num_feats_min, num_down=num_downs, 
                         max_channels=num_feats_max, norm=norm, 
                         use_dropout=False, outermost_linear=True,
                         upsampling_mode='transpose')
        
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
    def __init__(self,input_channel=3,output_channel=2,hidden_dim=24,block_num=6) -> None:
        super().__init__()
        self.input_channel=input_channel
        self.hidden_dim = hidden_dim
        self.block_num = block_num
        self.output_channel=output_channel
        self.input_layer=nn.Sequential(
            nn.Conv2d(self.input_channel,self.hidden_dim,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )
        self.middle_layer=self.make_layer(self.hidden_dim, self.hidden_dim, block_num=self.block_num)
        self.output_layer=nn.Sequential(
            nn.Conv2d(self.input_channel+self.hidden_dim,self.output_channel,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU()
        )
    
    def forward(self,x):
        identity = x
        x = self.input_layer(x)
        out=self.middle_layer(x)
        out = torch.cat((identity,out),dim=1) # concat channel
        out=self.output_layer(out)
        return out
    
    def make_layer(self, input_channel, output_channel, block_num,stride=1):
        layers=[]
        layers.append(residual_block(input_channel,output_channel))
        for _ in range(1, block_num):
            layers.append(residual_block(output_channel,output_channel))
        return nn.Sequential(*layers)
    
##############################
# Complex ResNet Propagation #
##############################

# class residual_block(nn.Module):
#     def __init__(self,input_channel, output_channel, stride=1, downsample=None) -> None:
#         super().__init__()
#         self. downsample=downsample
#         self.conv1=ComplexConv2d(input_channel,output_channel,kernel_size=3,stride=stride,padding=1,bias=False)
#         self.bn1=NaiveComplexBatchNorm2d(output_channel)
#         self.relu1=ComplexReLU()
#         self.conv2=ComplexConv2d(output_channel,output_channel,kernel_size=3,stride=stride,padding=1,bias=False)
#         self.bn2=NaiveComplexBatchNorm2d(output_channel)
#         self.relu2=ComplexReLU()
    
    
#     def forward(self,x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out = self.relu1(self.bn1(self.conv1(x)))
#         # out = self.bn2(self.conv2(out))
#         out = self.relu2(self.bn2(self.conv2(out)))
#         out = out.clone() + identity
#         return out

# class ResNet_Prop(nn.Module):
#     def __init__(self,input_channel=3,output_channel=2,hidden_dim=24,block_num=6) -> None:
#         super().__init__()
#         self.input_channel=input_channel
#         self.hidden_dim = hidden_dim
#         self.block_num = block_num
#         self.output_channel=output_channel
#         self.input_layer=nn.Sequential(
#             ComplexConv2d(self.input_channel,self.hidden_dim,kernel_size=3,padding=1,bias=False),
#             NaiveComplexBatchNorm2d(self.hidden_dim),
#             ComplexReLU()
#         )
#         self.middle_layer=self.make_layer(self.hidden_dim, self.hidden_dim, block_num=self.block_num)
#         self.output_layer=nn.Sequential(
#             ComplexConv2d(self.input_channel+self.hidden_dim,self.output_channel,kernel_size=3,padding=1,bias=False),
#             NaiveComplexBatchNorm2d(self.output_channel),
#             ComplexReLU()
#         )
    
#     def forward(self,x):
#         identity = x
#         x = self.input_layer(x)
#         out=self.middle_layer(x)
#         out = torch.cat((identity,out),dim=1) # concat channel
#         out=self.output_layer(out)
#         return out
    
#     def make_layer(self, input_channel, output_channel, block_num,stride=1):
#         layers=[]
#         layers.append(residual_block(input_channel,output_channel))
#         for _ in range(1, block_num):
#             layers.append(residual_block(output_channel,output_channel))
#         return nn.Sequential(*layers)

####################################################
# Different Implementations of Inverse Propagation #
####################################################

class InversePropagation(nn.Module):
    def __init__(self, inverse_network_config, **kwargs):
        super().__init__()
        self.config = inverse_network_config
        if self.config == 'cnn_only':
            self.inverse_cnn = ResNet_Prop(input_channel=len(kwargs['prop_dists_from_wrp']), output_channel=1, block_num=30)
            # self.inverse_cnn = UNetProp(img_size=kwargs['image_res'], input_nc=len(kwargs['prop_dists_from_wrp']), output_nc=1)
            self.forward = self.inverse_CNN_only
        elif self.config == 'cnn_asm_dpac':
            self.target_cnn = ResNet_Prop(input_channel=len(kwargs['prop_dists_from_wrp']), output_channel=2)
            self.asm_dpac = DPAC(kwargs['prop_dist'], kwargs['wavelength'], kwargs['feature_size'], prop_model='ASM', 
                                 propagator=propagation_ASM, device=kwargs['device'])
            self.forward = self.inverse_CNN_ASM_DPAC
        elif self.config == 'cnn_asm_cnn':
            self.target_cnn = UNetProp(img_size=kwargs['image_res'], input_nc=len(kwargs['prop_dists_from_wrp']), output_nc=2, num_downs=4)
            # self.target_cnn = ResNet_Prop(input_channel=len(kwargs['prop_dists_from_wrp']), output_channel=2, block_num=4)
            self.inverse_asm = prop_ideal.SerialProp(-kwargs['prop_dist'], kwargs['wavelength'], kwargs['feature_size'],
                                                     'ASM', kwargs['F_aperture'], None, dim=1)
            # self.slm_cnn = ResNet_Prop(input_channel=2, output_channel=1, block_num=6)
            self.slm_cnn = UNetProp(img_size=kwargs['image_res'], input_nc=2, output_nc=2, num_downs=4)
            self.forward = self.inverse_CNN_ASM_CNN
        elif self.config == 'cnn_asm_cnn_complex':
            self.target_cnn = ResNet_Prop(1, output_channel=1, block_num=3)
            self.inverse_asm = prop_ideal.SerialProp(-kwargs['prop_dist'], kwargs['wavelength'], kwargs['feature_size'],
                                        'ASM', kwargs['F_aperture'], None,
                                        dim=1)
            # self.slm_cnn = ResNet_Prop(input_channel=1, output_channel=1, block_num=4)
            self.slm_cnn = UNetProp(img_size=kwargs['image_res'], input_nc=2, output_nc=1, num_downs=4)
            self.forward = self.inverse_complex_CNN_ASM_CNN
        elif self.config == 'vit_only':
            self.uformer = Uformer(img_size=kwargs['image_res'][0], embed_dim=32, win_size=8, token_projection='linear',
                                   token_mlp='leff', modulator=True, dd_in=len(kwargs['prop_dists_from_wrp']), in_chans=1)
            self.forward = self.inverse_VIT_only
        # elif self.config == 'vit_2d':
        #     self.uformer2d = Uformer(img_size=kwargs['image_res'][0], embed_dim=32, win_size=8, token_projection='linear',
        #                            token_mlp='leff', modulator=True, dd_in=1, in_chans=1)
        #     self.forward = self.inverse_VIT_2d
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

    # def inverse_CNN_ASM_CNN(self, masked_imgs):
    #     ########## phase generated by CNN+ASM+CNN ###############
    #     mid_amp_phase = self.target_cnn(masked_imgs)
    #     mid_amp = mid_amp_phase[:,0:1,:,:]
    #     mid_phase = mid_amp_phase[:,1:2,:,:]
    #     mid_field = torch.complex(mid_amp * torch.cos(mid_phase), mid_amp * torch.sin(mid_phase))
    #     slm_field = self.inverse_asm(mid_field)
    #     slm_phase = self.slm_cnn(torch.cat([slm_field.abs(), slm_field.angle()], dim=1))
    #     ##########################################################
    #     return slm_phase
    
    def inverse_CNN_ASM_CNN(self, masked_imgs):
        ########## phase generated by CNN+ASM+CNN ###############
        mid_real_imag = self.target_cnn(masked_imgs)
        mid_real = mid_real_imag[:,0:1,:,:]
        mid_imag = mid_real_imag[:,1:2,:,:]
        mid_field = torch.complex(mid_real, mid_imag)
        slm_field = self.inverse_asm(mid_field)
        # slm_phase = self.slm_cnn(torch.cat([slm_field.real, slm_field.imag], dim=1))
        slm_field_encode = self.slm_cnn(torch.cat([slm_field.real, slm_field.imag], dim=1))
        slm_real = slm_field_encode[:,0:1,:,:]
        slm_imag = slm_field_encode[:,1:2,:,:]
        
        use_complex = True
        if use_complex:
            slm_complex = torch.complex(slm_real, slm_imag)
            slm_amp = slm_complex.abs()
            slm_phase = slm_complex.angle()
        else:
            slm_amp = torch.sqrt(slm_real**2 + slm_imag**2)
            slm_phase = torch.atan2(slm_imag, slm_real)
        # slm_phase = torch.atan2(slm_field_encode[:,1:2,:,:], slm_field_encode[:,0:1,:,:]) + np.pi
        ##########################################################
        return slm_amp, slm_phase

    def inverse_complex_CNN_ASM_CNN(self, masked_imgs):
        target_field = torch.complex(masked_imgs, torch.zeros_like(masked_imgs))
        target_field = target_field.to(masked_imgs.device)
        mid_field = self.target_cnn(target_field)
        slm_field = self.inverse_asm(mid_field)
        slm_field = self.slm_cnn(slm_field)
        # slm_phase = slm_field.angle()
        return slm_phase


    def inverse_VIT_only(self, masked_imgs):
        ########## phase generated by VIT only ###################
        slm_phase = self.uformer(masked_imgs)
        ##########################################################
        return slm_phase


    # def inverse_VIT_2d(self, imgs):
    #     ########## phase generated by VIT 2d only ###################
    #     slm_phase = self.uformer2d(imgs)
    #     #############################################################
    #     return slm_phase



if __name__ == '__main__':
    # from torchsummary import summary
    # reverse_prop = UNetProp((540,960), input_nc=8, output_nc=1, num_downs=8)
    # reverse_prop = reverse_prop.cuda()
    # summary(reverse_prop, (8, 540, 960))

    from load_flying3d import FlyingThings3D_loader
    from torch.utils.data import DataLoader
    import cv2, os
    import numpy as np

    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    device = torch.device('cuda:0')
    image_res=(1080, 1920)
    

    # inverse_prop = InversePropagation(inverse_network_config = 'cnn_asm_cnn', 
    #                                   prop_dists_from_wrp=[-4.4*mm, -3.2*mm, -2.4*mm, -1.0*mm, 0.0*mm, 1.3*mm, 2.8*mm, 3.8*mm],
    #                                   prop_dist=6.0*cm,
    #                                   wavelength=523.0*nm,
    #                                   feature_size=8.0*um,
    #                                   device=inference_device,
    #                                   F_aperture=0.5,
    #                                   image_res=image_res)

    model_path = 'runs\\2023-12-15-18-22-34.858603-FlyingThings3D-cnn_asm_cnn-0.0001-ASM-1080_1920-880_1600-1_target_planes-in-focus-loss-8e-06-0.06-MSE\model\FlyingThings3D-cnn_asm_cnn-0.0001-ASM-1080_1920-880_1600-1_target_planes-in-focus-loss-8e-06-0.06-MSE_best_psnr.pth'
    inverse_prop = torch.load(model_path)
    inverse_prop = inverse_prop.to(device)
    inverse_prop.eval()

    prop_dist = 60.0*mm
    wavelength = 523.7*nm
    feature_size = (8.0*um, 8.0*um)
    asm_dpac = DPAC(prop_dist, wavelength, feature_size, prop_model='ASM', 
                    propagator=propagation_ASM, device=device)


    
    data_path = 'C:\\data\\flying3D'
    roi_res = (880, 1600)
    plane_idx = [4]
    virtual_depth_planes = [0.0, 0.08417508417508479, 0.14124293785310726, 0.24299599771297942, 0.3171856978085348, 0.4155730533683304, 0.5319148936170226, 0.6112104949314254]
    virtual_depth_planes = [virtual_depth_planes[idx] for idx in plane_idx]


    return_type = 'image_mask_id'
    random_seed = 10

    # train_loader = FlyingThings3D_loader(data_path=data_path,
    #                                      channel=1, image_res=image_res, roi_res=roi_res,
    #                                      virtual_depth_planes=virtual_depth_planes,
    #                                      return_type=return_type,
    #                                      random_seed=random_seed,
    #                                      # slice=(0,0.025),
    #                                      # slice=(0,0.1),
    #                                      slice=(0,0.025),
    #                                      )
    
    # val_loader = FlyingThings3D_loader(data_path=data_path,
    #                                    channel=1, image_res=image_res, roi_res=roi_res,
    #                                    virtual_depth_planes=virtual_depth_planes,
    #                                    return_type=return_type,
    #                                    random_seed=random_seed,
    #                                    # slice=(0.995,1),
    #                                    slice=(0.2,0.205),
    #                                    )
    
    # test_loader = FlyingThings3D_loader(data_path=data_path,
    #                                    channel=1, image_res=image_res, roi_res=roi_res,
    #                                    virtual_depth_planes=virtual_depth_planes,
    #                                    return_type=return_type,
    #                                    random_seed=random_seed,
    #                                    # slice=(0.995,1),
    #                                    slice=(0.4,0.405),
    #                                    )
    
    # test_dataloader = DataLoader(test_loader, batch_size=1)

    # for imgs_masks_id in test_dataloader:
        
    #     imgs, masks, imgs_id = imgs_masks_id
    #     imgs = imgs.to(device)
    #     masks = masks.to(device)
    #     masked_imgs = imgs * masks

    #     # inverse propagation
    #     slm_phase = inverse_prop(masked_imgs)
        
    #     img_name = imgs_id[0].replace("\\","_")
    #     cv2.imwrite(f'{img_name}_phase_plane1_inverted.png', utils.phasemap_8bit(slm_phase))
    #     # cv2.imwrite(f'{img_name}_image.png', ((imgs) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8))

    #     break

    img_folder_path = 'C:\\data\\calibration_data'
    img_names = os.listdir(img_folder_path)
    #load all images from the folder img_folder_path end with .png
    imgs = []
    for img_name in img_names:
        if img_name.endswith('.png'):
            img = cv2.imread(os.path.join(img_folder_path, img_name), cv2.IMREAD_ANYCOLOR)
            img = np.fliplr(img)
            if len(img.shape) > 2:
                # get the green channel if the input image is rgb
                img = img[:,:,1]
            imgs.append(img)
    imgs = torch.tensor(imgs).unsqueeze(1).float().to(device)
    imgs = imgs / 255.0
    
    for i in range(imgs.shape[0]):
        img = imgs[i:i+1,:,:,:]
        slm_amp, slm_phase = inverse_prop(img)
        # _, slm_phase = asm_dpac(img)
        # torch.save(slm_phase, f'phase/{img_names[i]}_phase_plane.pt')
        
        cv2.imwrite(f'phase/inverse1p/real_imag/{img_names[i]}_phase_inverted.png', utils.phasemap_8bit(slm_phase, inverted=True, add_pi=False, shift=0.0))
        cv2.imwrite(f'phase/inverse1p/real_imag/{img_names[i]}_phase_inverted_shift_{0.25}.png', utils.phasemap_8bit(slm_phase, inverted=True, add_pi=False, shift=0.25 * np.pi))
        cv2.imwrite(f'phase/inverse1p/real_imag/{img_names[i]}_phase_inverted_shift_{0.5}.png', utils.phasemap_8bit(slm_phase, inverted=True, add_pi=False, shift=0.5 * np.pi))
        cv2.imwrite(f'phase/inverse1p/real_imag/{img_names[i]}_phase_inverted_shift_{0.75}.png', utils.phasemap_8bit(slm_phase, inverted=True, add_pi=False, shift=0.75 * np.pi))
        cv2.imwrite(f'phase/inverse1p/real_imag/{img_names[i]}_phase_inverted_shift_{1}.png', utils.phasemap_8bit(slm_phase, inverted=True, add_pi=False, shift=1.0 * np.pi))
        cv2.imwrite(f'phase/inverse1p/real_imag/{img_names[i]}_phase_inverted_shift_{1.25}.png', utils.phasemap_8bit(slm_phase, inverted=True, add_pi=False, shift=1.25 * np.pi))
        cv2.imwrite(f'phase/inverse1p/real_imag/{img_names[i]}_phase_inverted_shift_{1.5}.png', utils.phasemap_8bit(slm_phase, inverted=True, add_pi=False, shift=1.5 * np.pi))
        cv2.imwrite(f'phase/inverse1p/real_imag/{img_names[i]}_phase_inverted_shift_{1.75}.png', utils.phasemap_8bit(slm_phase, inverted=True, add_pi=False, shift=1.75 * np.pi))        
        # cv2.imwrite(f'{img_names[i]}_image.png', ((img) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8))