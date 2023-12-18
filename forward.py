# read slm phase and generate masked images on target planes

import os
import torch, cv2, utils

from PIL import Image
from torchvision import transforms
import prop_ideal

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = 60.0*mm
wavelength = 5.230e-07 # 5.177e-07
feature_size = (8.0e-06, 8.0e-06) # (6.4e-06, 6.4e-06)
F_aperture = 0.5
prop_dists_from_wrp = [-4.4*mm, -3.2*mm, -2.4*mm, -1.0*mm, 0.0*mm, 1.3*mm, 2.8*mm, 3.8*mm]
# specify how many target planes used to compute loss here
# plane_idx = [0, 1, 2, 3, 4, 5, 6, 7]
plane_idx = [4]
prop_dists_from_wrp = [prop_dists_from_wrp[idx] for idx in plane_idx]
device = torch.device('cuda:0')


input_phase_path = 'phase\\inverse1p\\calib-circle.png_phase_inverted.png'
input_phase_tensor_path = 'phase\\USAF15b.png_phase_plane.pt'

input_phase = Image.open(input_phase_path)
transform = transforms.Compose([transforms.ToTensor()])
input_phase = transform(input_phase)
input_phase = (1-input_phase)*2.0*3.1415926
# input_phase = (input_phase)*2.0*3.1415926

# input_phase = torch.load(input_phase_tensor_path)

input_phase.to(device)

forward_prop_list = ['ASM', 'CNNpropCNN']
forward_prop_id = 0
forward_prop_config = forward_prop_list[forward_prop_id]

if forward_prop_config == 'ASM':
    ######################## use ASM as Forward Network ###############################
    forward_prop = prop_ideal.SerialProp(prop_dist, wavelength, feature_size,
                                        'ASM', F_aperture, prop_dists_from_wrp,
                                        dim=1)
    forward_prop = forward_prop.to(device)
    ###################################################################################

# elif forward_prop_config == 'CNNpropCNN':
#     #################### use CNNpropCNN as Forward Network ############################
#     from prop_model import CNNpropCNN_default
#     forward_prop = CNNpropCNN_default(image_res, roi_res)
#     if forward_prop == None:
#         raise ValueError('CNNpropCNN only support image resolution 1080*1920/512*512 and roi 960*1680/448*448')
#     forward_prop = forward_prop.to(device)
#     for param in forward_prop.parameters():
#         param.requires_grad = False
#     ###################################################################################

outputs_field = forward_prop(input_phase)
outputs_amp = outputs_field.abs()



a = 1