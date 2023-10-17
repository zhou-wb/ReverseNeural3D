import torch, os, random, utils, time
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# propagation network related 
from inverse3d_prop import InversePropagation
from prop_model import CNNpropCNN_default
import prop_ideal

# dataset related
from load_hypersim import hypersim_TargetLoader
from load_flying3d import FlyingThings3D_loader



########################
# 1. Optics Parameters #
########################

# distance between the reference(middle) plane and slm
prop_dist = 0.0044
# distance between the reference(middle) plane and all the 8 target planes 
prop_dists_from_wrp = [-0.0044, -0.0032000000000000006, -0.0024000000000000002, -0.0010000000000000005, 0.0, 0.0013, 0.0028000000000000004, 0.0037999999999999987]
# depth in diopter space (m^-1) to compute the masks for rgbd input
virtual_depth_planes = [0.0, 0.08417508417508479, 0.14124293785310726, 0.24299599771297942, 0.3171856978085348, 0.4155730533683304, 0.5319148936170226, 0.6112104949314254]
# specify how many target planes used to compute loss here
plane_idx = [0, 1, 2, 3, 4, 5, 6, 7]
# plane_idx = [4]
prop_dists_from_wrp = [prop_dists_from_wrp[idx] for idx in plane_idx]
virtual_depth_planes = [virtual_depth_planes[idx] for idx in plane_idx]
wavelength = 5.177e-07
feature_size = (6.4e-06, 6.4e-06)
F_aperture = 0.5



##########################
# 2. Training Parameters #
##########################

device = torch.device('cuda:0')
loss_fn = nn.MSELoss().to(device)
learning_rate = 1e-4
max_epoch = 100000

loss_type_list = ['in-focus-loss', 'focal-stack-loss']
loss_id = 0
loss_type_config = loss_type_list[loss_id]

if loss_type_config == 'in-focus-loss':
    return_type = 'image_mask_id'
elif loss_type_config == 'focal-stack-loss':
    return_type = 'image_mask_focalstack_id'
# If there are nan in output, consider enable this to debug
# torch.autograd.set_detect_anomaly(True)




#########################
# 3. Dataset Parameters #
#########################

batch_size = 1
dataset_list = ['Hypersim', 'FlyingThings3D', 'MitCGH']
dataset_id = 1
dataset_name = dataset_list[dataset_id]
# loss_on_roi = True
# resize_to_1080p = False
# for_uformer = True
image_res = (512, 512)
roi_res = (448, 448)

random_seed = 10 #random_seed = None for not shuffle

if dataset_name == 'Hypersim':
    ############################# use Hypersim Dataset ##########################################
    # Load training set and validation set seperatly.
    # Set the slice parameter accordingly to get desired size of train/val sets
    # Original Resolution: (768, 1024), recommand roi: (680, 900)
    dir_list = ['ai_001_001','ai_004_001','ai_007_001','ai_010_001','ai_013_001','ai_019_001','ai_041_001','ai_050_001',
                'ai_002_001','ai_005_001','ai_008_001','ai_011_001','ai_014_003','ai_017_001','ai_021_001',
                'ai_003_001','ai_006_001','ai_009_001','ai_012_001','ai_015_001','ai_018_001','ai_030_001']
    data_path = ['/media/datadrive/hypersim/'+ folder + '/images' for folder in dir_list]
    # data_path = '/media/datadrive/hypersim/ai_001_001/images'
    
    train_loader = hypersim_TargetLoader(data_path=data_path, 
                                        channel=1, image_res=image_res, roi_res=roi_res,
                                        virtual_depth_planes=virtual_depth_planes,
                                        return_type=return_type,
                                        random_seed=random_seed,
                                        slice=(0,0.1),
                                        )
    val_loader = hypersim_TargetLoader(data_path=data_path, 
                                    channel=1, image_res=image_res, roi_res=roi_res,
                                    virtual_depth_planes=virtual_depth_planes,
                                    return_type=return_type,
                                    random_seed=random_seed,
                                    slice=(0.96,1),
                                    )
    #############################################################################################

elif dataset_name == 'FlyingThings3D':
    ######################### use FlyingThings3D Dataset ########################################
    # Load training set and validation set seperatly.
    # Set the slice parameter accordingly to get desired size of train/val sets
    # Original Resolution: (540, 960)
    # data_path = '/media/datadrive/flying3D'
    data_path = 'D:\\data\\flying3D'
    # if resize_to_1080p:
    #     image_res = (1080, 1920)
    #     if loss_on_roi:
    #         roi_res = (960, 1680)
    #     else:
    #         roi_res = (1080, 1920)
    # else:
    #     image_res = (540, 960)
    #     if loss_on_roi:
    #         roi_res = (480, 840)
    #     else:
    #         roi_res = (540, 960)
    # if for_uformer:
    #     image_res = (512, 512)
    #     roi_res = (448, 448)
        # image_res = (256, 256)
        # roi_res = (224, 224)

    train_loader = FlyingThings3D_loader(data_path=data_path,
                                         channel=1, image_res=image_res, roi_res=roi_res,
                                         virtual_depth_planes=virtual_depth_planes,
                                         return_type=return_type,
                                         random_seed=random_seed,
                                         # slice=(0,0.025),
                                         # slice=(0,0.1),
                                         slice=(0,0.025),
                                         )
    val_loader = FlyingThings3D_loader(data_path=data_path,
                                       channel=1, image_res=image_res, roi_res=roi_res,
                                       virtual_depth_planes=virtual_depth_planes,
                                       return_type=return_type,
                                       random_seed=random_seed,
                                       # slice=(0.995,1),
                                       slice=(0.2,0.205),
                                       )
    #############################################################################################
else:
    raise ValueError(f"Dataset: '{dataset_name}' Not Implement!")
    
# check the size of the training set and validation set
print(f"train set length: {len(train_loader)}")
print(f"val  set length: {len(val_loader)}")

# specify batch size here
train_dataloader = DataLoader(train_loader, batch_size=batch_size)
val_dataloader = DataLoader(val_loader, batch_size=batch_size)




#######################################
# 4. Load Networks -- Inverse Network #
#######################################

# choose the network structure by set the config_id to 0,1,2
inverse_network_list = ['cnn_only', 'cnn_asm_dpac', 'cnn_asm_cnn', 'vit_only']
network_id = 3
inverse_network_config = inverse_network_list[network_id]

inverse_prop = InversePropagation(inverse_network_config, prop_dists_from_wrp=prop_dists_from_wrp, prop_dist=prop_dist,
                                  wavelength=wavelength, feature_size=feature_size, device=device, F_aperture=F_aperture,
                                  image_res=image_res)

inverse_prop = inverse_prop.to(device)
optimizer = torch.optim.Adam(inverse_prop.parameters(), lr=learning_rate)


#######################################
# 5. Load Networks -- Forward Network #
#######################################

forward_prop_list = ['ASM', 'CNNpropCNN']
forward_prop_id = 1
forward_prop_config = forward_prop_list[forward_prop_id]

if forward_prop_config == 'ASM':
    ######################## use ASM as Forward Network ###############################
    forward_prop = prop_ideal.SerialProp(prop_dist, wavelength, feature_size,
                                        'ASM', F_aperture, prop_dists_from_wrp,
                                        dim=1)
    forward_prop = forward_prop.to(device)
    ###################################################################################

elif forward_prop_config == 'CNNpropCNN':
    #################### use CNNpropCNN as Forward Network ############################
    forward_prop = CNNpropCNN_default(image_res, roi_res)
    if forward_prop == None:
        raise ValueError('CNNpropCNN only support image resolution 1080*1920/512*512 and roi 960*1680/448*448')
    forward_prop = forward_prop.to(device)
    for param in forward_prop.parameters():
        param.requires_grad = False
    ###################################################################################






################
# Init metrics #
################

total_train_step = 0
best_val_loss = float('inf')
best_test_psnr = 0

# init tensorboard
time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
run_id = dataset_name + '-' + inverse_network_config + '-' + \
    str(learning_rate) + '-' + forward_prop_config + '-' + \
    f'{image_res[0]}_{image_res[1]}-{roi_res[0]}_{roi_res[1]}' + '-' + \
    f'{len(plane_idx)}_target_planes' + '-' + loss_type_config
print('Dataset:', dataset_name)
print('Inverse Network:', inverse_network_config)
print('Forward Prop:', forward_prop_config)
print('Learning Rate:', learning_rate)
print('Image Resolution:', image_res)
print('ROI Resolution:', roi_res)
print('Batch Size:', batch_size)
print('Number of Target Planes:', len(plane_idx))
print('Loss Type:', loss_type_config)
print('The Network will be Trained on:', torch.cuda.get_device_name(device))
input("Press Enter to continue...")
run_folder_name = time_str + '-' + run_id
writer = SummaryWriter(f'runs/{run_folder_name}')
writer.add_scalar("learning_rate", learning_rate)




#################
# Training loop #
#################

for i in range(max_epoch):
    print(f"----------Training Start (Epoch: {i+1})-------------")
    total_train_loss, total_train_psnr = 0, 0
    train_items_count = 0
    # training steps
    inverse_prop.train()
    average_scale_factor = 0
    for imgs_masks_id in train_dataloader:
        
        if loss_type_config == 'in-focus-loss':
            imgs, masks, imgs_id = imgs_masks_id
        elif loss_type_config == 'focal-stack-loss':
            imgs, masks, focalstack, imgs_id = imgs_masks_id
            focalstack = focalstack.to(device)

        imgs = imgs.to(device)
        masks = masks.to(device)
        masked_imgs = imgs * masks

        # inverse propagation
        slm_phase = inverse_prop(masked_imgs)
        # forward propagation
        outputs_field = forward_prop(slm_phase)
        outputs_amp = outputs_field.abs()
        
        imgs = utils.crop_image(imgs, roi_res, stacked_complex=False)
        masks = utils.crop_image(masks, roi_res, stacked_complex=False) # need to check if process before network
        outputs_amp = utils.crop_image(outputs_amp, roi_res, stacked_complex=False)
        
        if loss_type_config == 'in-focus-loss':
            nonzeros = masks > 0
            final_amp = torch.zeros_like(outputs_amp)
            final_amp[nonzeros] += (outputs_amp[nonzeros] * masks[nonzeros])
            masked_imgs = utils.crop_image(masked_imgs, roi_res, stacked_complex=False)

            with torch.no_grad():
                # s = (final_amp * masked_imgs).mean() / \
                #     (final_amp ** 2).mean()  # scale minimizing MSE btw recon and target
                s = 1
                average_scale_factor += s
            loss = loss_fn(s * final_amp, masked_imgs)
        
        elif loss_type_config == 'focal-stack-loss':
            focalstack = utils.crop_image(focalstack, roi_res, stacked_complex=False)
        
            with torch.no_grad():
                s = (outputs_amp * focalstack).mean() / \
                    (outputs_amp ** 2).mean()  # scale minimizing MSE btw recon and target
                average_scale_factor += s
            loss = loss_fn(s * outputs_amp, focalstack)
        
        writer.add_scalar("scale", s, total_train_step)
        
        with torch.no_grad(): 
            psnr = utils.calculate_psnr(utils.target_planes_to_one_image(s * outputs_amp, masks), imgs[:,0])
        
        total_train_loss += loss.item()
        total_train_psnr += psnr.item()
        writer.add_scalar("train_loss", loss.item(), total_train_step)
        writer.add_scalar("train_psnr", psnr.item(), total_train_step)
        
        # optimization
        optimizer.zero_grad()
        loss.backward()
        # print(masked_imgs.grad)
        # print(mid_amp_phase.grad)
        optimizer.step()
        
        total_train_step = total_train_step + 1
        train_items_count += 1
        if (total_train_step) % 100 == 0:
            
            print(f"Training Step {total_train_step}, Loss: {loss.item()}")
            
            if (total_train_step) % 1000 == 0:
                # mapped_slm_phase = ((slm_phase + np.pi) % (2 * np.pi)) / (2 * np.pi)
                mapped_slm_phase = ((slm_phase - slm_phase.min()) / (slm_phase.max() - slm_phase.min()))
                writer.add_text('image id', imgs_id[0], total_train_step)
                writer.add_image(f'phase', mapped_slm_phase[0,0], total_train_step, dataformats='HW')
                for i in range(len(plane_idx)):
                    writer.add_image(f'input_images_plane{i}', masked_imgs[0,i], total_train_step, dataformats='HW')
                    writer.add_images(f'output_image_plane{i}', outputs_amp[0,i], total_train_step, dataformats='HW')
                    # writer.add_image(f'input_images_plane{i}', masked_imgs.squeeze()[i,:,:], total_train_step, dataformats='CHW')
                    # writer.add_image(f'output_image_plane{i}', outputs_amp.squeeze()[i,:,:], total_train_step, dataformats='CHW')
                writer.flush()    
    
    average_train_loss = total_train_loss/train_items_count
    average_train_psnr = total_train_psnr/train_items_count
    average_scale_factor = average_scale_factor/train_items_count
    writer.add_scalar("average_train_loss", average_train_loss, total_train_step)
    writer.add_scalar("average_train_psnr", average_train_psnr, total_train_step)
    writer.add_scalar("average_scale_factor", average_scale_factor, total_train_step)
    
    ###################
    # Validation loop #
    ###################
    # test the model on validation set after every epoch
    inverse_prop.eval()
    total_val_loss = 0
    total_val_psnr = 0
    val_items_count = 0
    record_id = random.randrange(len(val_loader)//batch_size)
    # record_id = 0
    with torch.no_grad():
        for imgs_masks_id in val_dataloader: 
            if loss_type_config == 'in-focus-loss':
                imgs, masks, imgs_id = imgs_masks_id
            elif loss_type_config == 'focal-stack-loss':
                imgs, masks, focalstack, imgs_id = imgs_masks_id
                focalstack = focalstack.to(device)

            imgs = imgs.to(device)
            masks = masks.to(device)
            masked_imgs = imgs * masks
            
            # inverse propagation
            slm_phase = inverse_prop(masked_imgs)
            # forward propagation
            outputs_field = forward_prop(slm_phase)
            outputs_amp = outputs_field.abs()
            
            imgs = utils.crop_image(imgs, roi_res, stacked_complex=False)
            masks = utils.crop_image(masks, roi_res, stacked_complex=False) # need to check if process before network
            outputs_amp = utils.crop_image(outputs_amp, roi_res, stacked_complex=False)
            
            if loss_type_config == 'in-focus-loss':
                nonzeros = masks > 0
                final_amp = torch.zeros_like(outputs_amp)
                final_amp[nonzeros] += (outputs_amp[nonzeros] * masks[nonzeros])
                masked_imgs = utils.crop_image(masked_imgs, roi_res, stacked_complex=False)
                
                loss = loss_fn(average_scale_factor * final_amp, masked_imgs)
            
            elif loss_type_config == 'focal-stack-loss':
                focalstack = utils.crop_image(focalstack, roi_res, stacked_complex=False)
            
                loss = loss_fn(average_scale_factor * outputs_amp, focalstack)
            
            
            with torch.no_grad(): 
                psnr = utils.calculate_psnr(utils.target_planes_to_one_image(average_scale_factor * outputs_amp, masks), imgs[:,0])
            
            if val_items_count == record_id:
                # mapped_slm_phase = ((slm_phase + np.pi) % (2 * np.pi)) / (2 * np.pi)
                mapped_slm_phase = ((slm_phase - slm_phase.min()) / (slm_phase.max() - slm_phase.min()))
                writer.add_text('val_image id', imgs_id[0], total_train_step)
                writer.add_image(f'val_phase', mapped_slm_phase[0,0], total_train_step, dataformats='HW')
                for i in range(len(plane_idx)):
                    writer.add_image(f'val_input_images_plane{i}', masked_imgs[0,i], total_train_step, dataformats='HW')
                    writer.add_images(f'val_output_image_plane{i}', outputs_amp[0,i], total_train_step, dataformats='HW')
                    # writer.add_image(f'input_images_plane{i}', masked_imgs.squeeze()[i,:,:], total_train_step, dataformats='CHW')
                    # writer.add_image(f'output_image_plane{i}', outputs_amp.squeeze()[i,:,:], total_train_step, dataformats='CHW')
                writer.flush()
            
            total_val_loss += loss.item()
            total_val_psnr += psnr.item()
            val_items_count += 1
        
        average_val_loss = total_val_loss/val_items_count
        average_val_psnr = total_val_psnr/val_items_count
        if best_val_loss > average_val_loss:
            best_val_loss = average_val_loss
            # save model
            path = f"runs/{run_folder_name}/model/"
            if not os.path.exists(path):
                os.makedirs(path) 
            torch.save(inverse_prop, f"{path}/{run_id}_best_loss.pth")
            writer.add_scalar("best_scale_factor", average_scale_factor, total_train_step)
            print("model saved!")
            
    print(f"Average Val Loss: {average_val_loss}")
    print(f"Average Val PSNR: {average_val_psnr}")
    writer.add_scalar("average_val_loss", average_val_loss, total_train_step)
    writer.add_scalar("average_val_psnr", average_val_psnr, total_train_step)
