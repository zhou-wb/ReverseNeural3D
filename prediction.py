import torch
from torch.utils.data import DataLoader
from inverse3d_prop import UNetProp, ResNet_Prop, InversePropagation
from prop_model import CNNpropCNN_default
import prop_ideal
from propagation_ASM import propagation_ASM
from algorithm import DPAC
from load_hypersim import hypersim_TargetLoader
from load_flying3d import FlyingThings3D_loader
import utils
import time
from torch import nn


#####################
# Optics Parameters #
#####################

# distance between the reference(middle) plane and slm
prop_dist = 0.0044
# distance between the reference(middle) plane and all the 8 target planes 
prop_dists_from_wrp = [-0.0044, -0.0032000000000000006, -0.0024000000000000002, -0.0010000000000000005, 0.0, 0.0013, 0.0028000000000000004, 0.0037999999999999987]
# depth in diopter space (m^-1) to compute the masks for rgbd input
virtual_depth_planes = [0.0, 0.08417508417508479, 0.14124293785310726, 0.24299599771297942, 0.3171856978085348, 0.4155730533683304, 0.5319148936170226, 0.6112104949314254]
# specify how many target planes used to compute loss here
plane_idx = [0, 1, 2, 3, 4, 5, 6, 7]
prop_dists_from_wrp = [prop_dists_from_wrp[idx] for idx in plane_idx]
virtual_depth_planes = [virtual_depth_planes[idx] for idx in plane_idx]
wavelength = 5.177e-07
feature_size = (6.4e-06, 6.4e-06)
F_aperture = 0.5



#######################
# Training Parameters #
#######################

device = torch.device('cuda:0')
loss_fn = nn.MSELoss().to(device)
learning_rate = 1e-2
max_epoch = 100000
# If there are nan in output, consider enable this to debug
# torch.autograd.set_detect_anomaly(True)




######################
# Dataset Parameters #
######################

batch_size = 1
dataset_list = ['Hypersim', 'FlyingThings3D', 'MitCGH']
dataset_id = 1
dataset_name = dataset_list[dataset_id]
loss_on_roi = True
resize_to_1080p = True
random_seed = 10 #random_seed = None for not shuffle


if dataset_name == 'Hypersim':
    ############################# use Hypersim Dataset ##########################################
    # Load training set and validation set seperatly.
    # Set the slice parameter accordingly to get desired size of train/val sets
    # Original Resolution: (768, 1024)
    dir_list = ['ai_001_001','ai_004_001','ai_007_001','ai_010_001','ai_013_001','ai_019_001','ai_041_001','ai_050_001',
                'ai_002_001','ai_005_001','ai_008_001','ai_011_001','ai_014_003','ai_017_001','ai_021_001',
                'ai_003_001','ai_006_001','ai_009_001','ai_012_001','ai_015_001','ai_018_001','ai_030_001']
    data_path = ['/media/datadrive/hypersim/'+ folder + '/images' for folder in dir_list]
    # data_path = '/media/datadrive/hypersim/ai_001_001/images'

    if resize_to_1080p:
        image_res = (1080, 1920)
        if loss_on_roi:
            roi_res = (960, 1680)
        else:
            roi_res = (1080, 1920)
    else:
        image_res = (768, 1024)
        if loss_on_roi:
            roi_res = (680, 900)
        else:
            roi_res = (768, 1024)
    train_loader = hypersim_TargetLoader(data_path=data_path, 
                                        channel=1, image_res=image_res, roi_res=roi_res,
                                        virtual_depth_planes=virtual_depth_planes,
                                        return_type='image_mask_id',
                                        random_seed=random_seed,
                                        slice=(0,0.1),
                                        )
    val_loader = hypersim_TargetLoader(data_path=data_path, 
                                    channel=1, image_res=image_res, roi_res=roi_res,
                                    virtual_depth_planes=virtual_depth_planes,
                                    return_type='image_mask_id',
                                    random_seed=random_seed,
                                    slice=(0.96,1),
                                    )
    #############################################################################################

elif dataset_name == 'FlyingThings3D':
    ######################### use FlyingThings3D Dataset ########################################
    # Load training set and validation set seperatly.
    # Set the slice parameter accordingly to get desired size of train/val sets
    # Original Resolution: (540, 960)
    data_path = './media/datadrive/flying3D'
    if resize_to_1080p:
        image_res = (1080, 1920)
        if loss_on_roi:
            roi_res = (960, 1680)
        else:
            roi_res = (1080, 1920)
    else:
        image_res = (540, 960)
        if loss_on_roi:
            roi_res = (480, 840)
        else:
            roi_res = (540, 960)
    train_loader = FlyingThings3D_loader(data_path=data_path,
                                         channel=1, image_res=image_res, roi_res=roi_res,
                                         virtual_depth_planes=virtual_depth_planes,
                                         return_type='image_mask_id',
                                         random_seed=random_seed,
                                         # slice=(0,0.025),
                                         # slice=(0,0.1),
                                         slice=(0,0.025),
                                         )
    val_loader = FlyingThings3D_loader(data_path=data_path,
                                       channel=1, image_res=image_res, roi_res=roi_res,
                                       virtual_depth_planes=virtual_depth_planes,
                                       return_type='image_mask_id',
                                       random_seed=random_seed,
                                       # slice=(0.995,1),
                                       slice=(0.2,0.205),
                                       )
    #############################################################################################
else:
    raise ValueError(f'Dataset: {dataset_name} Not Implement!')
    
# check the size of the training set and validation set
print(f"train set length: {len(train_loader)}")
print(f"val  set length: {len(val_loader)}")

# specify batch size here
train_dataloader = DataLoader(train_loader, batch_size=batch_size)
val_dataloader = DataLoader(val_loader, batch_size=batch_size)


# Load the saved model
# model_path = r'runs\2023-07-16-15-33-10.607579-FlyingThings3D-cnn_asm_cnn-0.01-ASM-540_960-480_840\model\FlyingThings3D-cnn_asm_cnn-0.01-ASM-540_960-480_840_best_loss.pth'
model_path = r'runs\2023-07-16-16-41-32.795857-FlyingThings3D-cnn_asm_cnn-0.01-ASM-1080_1920-960_1680\model\FlyingThings3D-cnn_asm_cnn-0.01-ASM-1080_1920-960_1680_best_loss.pth'
inverse_prop = torch.load(model_path)
inverse_prop.eval()  # switch the model to evaluation mode





####################################
# Load Networks -- Forward Network #
####################################

#################### use CNNpropCNN as Forward Network ############################
# forward_network_config = 'CNNpropCNN'
# if resize_to_1080p == False:
#     raise ValueError('You MUST set resize_to_1080p to True to use CNNpropCNN as forward propagation model')
# forward_prop = CNNpropCNN_default()
# forward_prop = forward_prop.to(device)
# for param in forward_prop.parameters():
#     param.requires_grad = False
###################################################################################

######################## use ASM as Forward Network ###############################
forward_network_config = 'ASM'
forward_prop = prop_ideal.SerialProp(prop_dist, wavelength, feature_size,
                                     'ASM', F_aperture, prop_dists_from_wrp,
                                     dim=1)
forward_prop = forward_prop.to(device)
###################################################################################






# Inference loop
for imgs_masks_id in val_dataloader:
    imgs, masks, imgs_id = imgs_masks_id
    imgs = imgs.to(device)
    masks = masks.to(device)
    masked_imgs = imgs * masks
    masks = utils.crop_image(masks, roi_res, stacked_complex=False) # need to check if process before network
    nonzeros = masks > 0
    # Start the timer

    # Perform inference
    with torch.no_grad():

        start_time = time.time()
        slm_phase = inverse_prop(masked_imgs)
        elapsed_time = time.time() - start_time

        outputs_field = forward_prop(slm_phase)
        outputs_field = utils.crop_image(outputs_field, roi_res, stacked_complex=False)
        outputs_amp = outputs_field.abs()
        final_amp = torch.zeros_like(outputs_amp)
        final_amp[nonzeros] += (outputs_amp[nonzeros] * masks[nonzeros])


    # Stop the timer

    # Print out the elapsed time
    print(f'Time to generate prediction for image {imgs_id[0]}: {elapsed_time} seconds')

    # # Save the prediction to disk
    # prediction_path = 'path/to/save/predictions/'  # replace this with the path where you want to save predictions
    # torch.save(final_amp.cpu(), f'{prediction_path}/prediction_{imgs_id[0]}.pt')

