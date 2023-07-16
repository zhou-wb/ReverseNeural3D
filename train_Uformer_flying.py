import torch
import numpy as np
import matplotlib.pyplot as plt
from load_image import LSHMV_RGBD_Object_Dataset
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from reverse3d_prop import Reverse3dProp
from resnet_prop import ResNet_Prop
from torch import nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import utils
from prop_model import CNNpropCNN_default
import prop_ideal
import load_flying3d

from propagation_ASM import propagation_ASM
from algorithm import DPAC

from load_hypersim import hypersim_TargetLoader
from Uformer_model import Uformer


# torch.autograd.set_detect_anomaly(True)

prop_dist = 0.0044
prop_dists_from_wrp = [-0.0044, -0.0032000000000000006, -0.0024000000000000002, -0.0010000000000000005, 0.0, 0.0013, 0.0028000000000000004, 0.0037999999999999987]
virtual_depth_planes = [0.0, 0.08417508417508479, 0.14124293785310726, 0.24299599771297942, 0.3171856978085348, 0.4155730533683304, 0.5319148936170226, 0.6112104949314254]
plane_idx = [0, 1, 2, 3, 4, 5, 6, 7]
prop_dists_from_wrp = [prop_dists_from_wrp[idx] for idx in plane_idx]
virtual_depth_planes = [virtual_depth_planes[idx] for idx in plane_idx]
wavelength = 5.177e-07
feature_size = (6.4e-06, 6.4e-06)
F_aperture = 0.5

# if torch.cuda.is_available():
#     # 如果存在多个CUDA设备，选择cuda:1，否则使用cuda:0
#     device = torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
device = torch.device('cuda:0')

# img_dir = '/home/wenbin/Downloads/rgbd-scenes-v2/imgs/scene_01'

# image_res = (540, 960)
image_res = (512, 512)
# roi_res = (540, 960)
roi_res = (512, 512)

# image_res = (768, 1024)
# roi_res = (768, 1024)

# tf = transforms.Compose([
#     Resize(image_res),
#     ToTensor()
# ])
# tf = transforms.Resize(image_res)
tf = transforms.CenterCrop(image_res)

# nyu_dataset = LSHMV_RGBD_Object_Dataset(img_dir,
#                                         color_transform=tf, depth_transform=tf,
#                                         channel=1, output_type='mask')

# for i in range(2, 15):
#     scene_name = 'scene_' + str(i).zfill(2)
#     nyu_dataset += LSHMV_RGBD_Object_Dataset('/home/wenbin/Downloads/rgbd-scenes-v2/imgs/'+scene_name,
#                                              color_transform=tf, depth_transform=tf,
#                                              channel=1, output_type='mask')

# img_loader = hypersim_TargetLoader(data_path='/media/datadrive/hypersim/ai_001_001/images', 
#                             channel=1, 
#                             shuffle=False, 
#                             virtual_depth_planes=virtual_depth_planes,
#                             return_type='image_mask_id',
#                             )

img_loader = load_flying3d.FlyingThings3D_loader('./media/datadrive/flying3D',
                                        channel=1, 
                                        shuffle=False, 
                                        virtual_depth_planes=virtual_depth_planes,
                                        return_type='image_mask_id',
                                        )
    
    
# train_data_size = int(0.8*len(nyu_dataset))
# test_data_size = len(nyu_dataset)-train_data_size
# train_data, test_data = random_split(nyu_dataset, [train_data_size,test_data_size], generator=torch.Generator().manual_seed(42))

# print(f"train set length: {train_data_size}")
# print(f"test  set length: {test_data_size}")

train_dataloader = DataLoader(img_loader, batch_size=1)
# train_dataloader = DataLoader(img_loader, batch_size=1)
# test_dataloader = DataLoader(test_data, batch_size=1)

# reverse_prop = Reverse3dProp()
# reverse_prop = ResNet_Prop(input_channel=len(prop_dists_from_wrp))
depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
# depths=[4, 4, 4, 4, 4, 4, 4, 4, 4]
reverse_prop = Uformer(img_size=image_res, embed_dim=64,depths=depths, dd_in=len(prop_dists_from_wrp), in_chans=2,  # out_channel=in_chans
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)
reverse_prop = reverse_prop.to(device)


asm_dpac = DPAC(prop_dist, wavelength, feature_size, prop_model='ASM', propagator=propagation_ASM, device=device)

# for param in reverse_prop.CNNpropCNN.parameters():
#     param.requires_grad = False
forward_prop = prop_ideal.SerialProp(prop_dist, wavelength, feature_size,
                                     'ASM', F_aperture, prop_dists_from_wrp,
                                     dim=1)
forward_prop = forward_prop.to(device)
# for param in forward_prop.parameters():
#     param.requires_grad = False
asm_dpac = asm_dpac.to(device)
# for param in asm_dpac.parameters():
#     param.requires_grad = False

# # Mean Squared Error Loss
# mse_loss = nn.MSELoss().to(device)

# # Perceptual Loss: often we use a pre-trained model, here we use VGG16
# vgg = models.vgg16(pretrained=True)
# # We usually use the output of one of the middle layers as the perceptual loss
# # here we use the second max pooling layer
# vgg_layer = vgg.features[:5].eval().to(device)
# mse = nn.MSELoss().to(device)

# def perceptual_loss(img1, img2):
#     # zeros = torch.zeros_like(img1)
#     # img1 = torch.cat((img1, zeros), dim=1)
#     # img2 = torch.cat((img2, zeros), dim=1)
#     # mse_1 = mse(vgg_layer(img1[:,0:3,:,:]), vgg_layer(img2[:,0:3,:,:]))
#     # mse_2 = mse(vgg_layer(img1[:,3:6,:,:]), vgg_layer(img2[:,3:6,:,:]))
#     # mse_3 = mse(vgg_layer(img1[:,6:9,:,:]), vgg_layer(img2[:,6:9,:,:]))
#     # return mse_1 + mse_2 + mse_3

#     return mse(vgg_layer(img1), vgg_layer(img2))

# # Total Variation Loss
# def tv_loss(img):
#     return torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
#            torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

# # Combined Loss
# def combined_loss(img_pred, img_gt, weights={'mse': 1.0, 'perceptual': 0.025, 'tv': 0.001}):
#     return weights['mse'] * mse_loss(img_pred, img_gt) + \
#            weights['perceptual'] * perceptual_loss(img_pred, img_gt) + \
#            weights['tv'] * tv_loss(img_pred)
# loss_fn = combined_loss

loss_fn = nn.MSELoss()


learning_rate = 10e-6
optimizer = torch.optim.AdamW(reverse_prop.parameters(), lr=learning_rate, betas=(0.9, 0.999))
# optimizer = torch.optim.SGD(reverse_prop.parameters(), lr=learning_rate)

epoch = 10

total_train_step = 0
best_test_loss = float('inf')
best_test_psnr = 0

# 添加tensorboard
time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
# writer = SummaryWriter("./logs")
writer = SummaryWriter(f'runs/{time_str}')

writer.add_scalar("learning_rate", learning_rate)
# torch.autograd.set_detect_anomaly(True)

for i in range(epoch):
    print(f"----------Training Start (Epoch: {i+1})-------------")
    total_train_loss, total_train_psnr = 0, 0
    
    # training steps
    reverse_prop.train()
    for imgs_masks_id in train_dataloader:
        imgs, masks, imgs_id = imgs_masks_id
        # imgs = imgs.to(device)
        imgs = tf(imgs).to(device)
        # masks = masks.to(device)
        masks = tf(masks).to(device)
        masked_imgs = imgs * masks
        nonzeros = masks > 0
        masks = utils.crop_image(masks, roi_res, stacked_complex=False) # need to check if process before network
        # outputs_field = reverse_prop(imgs)
        # outputs_amp = outputs_field.abs()
        # final_amp = outputs_amp*masks
        mid_amp_phase = reverse_prop(masked_imgs) # torch.Size([1, 2, 540, 960])
        # print('mid_amp_phase.shape:',mid_amp_phase.shape)
        
        mid_amp = mid_amp_phase[:,0:1,:,:] # torch.Size([1, 1, 512, 512])
        mid_phase = mid_amp_phase[:,1:2,:,:] # torch.Size([1, 1, 512, 512])
        # print('mid_amp.shape:',mid_amp.shape)
        # print('mid_phase.shape:',mid_phase.shape)
        
        # mid_amp = (mid_amp-mid_amp.min())/(mid_amp.max()-mid_amp.min())
        # _, slm_phase = asm_dpac(mid_amp, mid_phase)
        
        _, slm_phase = asm_dpac(mid_amp, mid_phase)
        
        # slm_phase = mid_amp_phase[:,1:2,:,:]
        outputs_field = forward_prop(slm_phase)
        
        
        outputs_field = utils.crop_image(outputs_field, roi_res, stacked_complex=False)
        
        outputs_amp = outputs_field.abs()
        final_amp = outputs_amp*masks # [1 ,8, 512, 512]
        # final_amp = torch.zeros_like(outputs_amp)
        # final_amp[nonzeros] += (outputs_amp[nonzeros] * masks[nonzeros])
        
        masked_imgs = utils.crop_image(masked_imgs, roi_res, stacked_complex=False) # need to check if process before network or only before loss 
        
        with torch.no_grad():
            s = (final_amp * masked_imgs).mean() / \
                (final_amp ** 2).mean()  # scale minimizing MSE btw recon and target
        # loss = loss_fn(s * final_amp, masked_imgs)

        # img_pred = torch.sum(s * final_amp, dim=1, keepdim=True) # [1 ,1, 512, 512]

        # plt.figure()
        # plt.subplot(1, 2, 1)  # 参数分别表示：行数，列数，子图的索引
        # plt.imshow(img_pred[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(imgs[0, 0, :, :].cpu().detach().numpy(), cmap='gray')
        # # plt.colorbar()
        # plt.show()

        # Green层前后拼接0层
        # zeros = torch.zeros_like(img_pred)
        # zeros_img_pred_zeros = torch.cat((zeros, img_pred, zeros), dim=1) # [1, 3, 512, 512]
        # zeros_img_gt_zeros = torch.cat((zeros, imgs, zeros), dim=1) # [1, 3, 512, 512]
        # loss = loss_fn(zeros_img_pred_zeros, zeros_img_gt_zeros)
        
        loss = loss_fn(final_amp, masked_imgs)
        
        # a = utils.target_planes_to_one_image(final_amp, masks)
        # b = utils.target_planes_to_one_image(imgs, masks)
        
        with torch.no_grad(): 
            psnr = utils.calculate_psnr(utils.target_planes_to_one_image(s * final_amp, masks), utils.target_planes_to_one_image(imgs, masks))
        
        total_train_loss += loss.item()
        total_train_psnr += psnr

        writer.add_scalar("train_loss", loss.item(), total_train_step)
        writer.add_scalar("train_psnr", psnr.item(), total_train_step)
        
        # optimization
        optimizer.zero_grad()
        loss.backward()
        # print(masked_imgs.grad)
        # print(mid_amp_phase.grad)
        optimizer.step()
        
        total_train_step = total_train_step + 1
        if (total_train_step) % 100 == 0:
            
            print(f"Training Step {total_train_step}, Loss: {loss.item()}")
            
            if (total_train_step) % 1000 == 0:
                mapped_slm_phase = ((slm_phase + np.pi) % (2 * np.pi)) / (2 * np.pi)
                writer.add_text('image id', imgs_id[0], total_train_step)
                writer.add_image(f'phase', mapped_slm_phase.squeeze(), total_train_step, dataformats='HW')
                for i in range(len(plane_idx)):
                    writer.add_image(f'input_images_plane{i}', masked_imgs.squeeze()[i,:,:], total_train_step, dataformats='HW')
                    writer.add_images(f'output_image_plane{i}', outputs_amp.squeeze()[i,:,:], total_train_step, dataformats='HW')
                    # writer.add_image(f'input_images_plane{i}', masked_imgs.squeeze()[i,:,:], total_train_step, dataformats='CHW')
                    # writer.add_image(f'output_image_plane{i}', outputs_amp.squeeze()[i,:,:], total_train_step, dataformats='CHW')
                    writer.flush()
    
    # average_train_loss, average_train_psnr = (total_train_loss, total_train_psnr)/train_dataloader.len()
    # writer.add_scalar("average_train_loss", average_train_loss, total_train_step)
    # writer.add_scalar("average_train_psnr", average_train_psnr, total_train_step)
        
    # test the model after every epoch
    # reverse_prop.eval()
    # total_test_loss = 0
    # total_test_psnr = 0
    # test_items_count = 0
    # with torch.no_grad():
    #     for imgs_masks_id in test_dataloader:
    #         imgs, masks, imgs_id = imgs_masks_id
    #         imgs = imgs.to(device)
    #         masks = masks.to(device)
    #         # outputs_field = reverse_prop(imgs)
    #         # outputs_amp = outputs_field.abs()
    #         # final_amp = outputs_amp * masks
            
    #         slm_phase = reverse_prop(imgs)
    #         outputs_field = forward_prop(slm_phase)
    #         outputs_amp = outputs_field.abs()
    #         final_amp = outputs_amp*masks
            
    #         # outputs = reverse_prop(imgs)
    #         loss = loss_fn(final_amp, imgs)
    #         psnr = utils.calculate_psnr(utils.target_planes_to_one_image(final_amp, masks), utils.target_planes_to_one_image(imgs, masks))
            
    #         total_test_loss += loss
    #         total_test_psnr += psnr
    #         test_items_count += 1
        
    #     average_test_loss = total_test_loss/test_items_count
    #     average_test_psnr = total_test_psnr/test_items_count
    #     if best_test_loss > average_test_loss:
    #         best_test_loss = average_test_loss
    #         # save model
    #         path = f"runs/{time_str}/model/"
    #         if not os.path.exists(path):
    #             os.makedirs(path) 
    #         torch.save(reverse_prop, f"runs/{time_str}/model/reverse_3d_prop_{time_str}_best_loss.pth")
    #         print("model saved!")
            
    # print(f"Average Test Loss: {average_test_loss}")
    # print(f"Average Test PSNR: {average_test_psnr}")
    # writer.add_scalar("average_test_loss", average_test_loss.item(), total_train_step)
    # writer.add_scalar("average_test_psnr", average_test_psnr.item(), total_train_step)
    
    

# with torch.no_grad():
#     s = (final_amp * target_amp).mean() / \
#         (final_amp ** 2).mean()  # scale minimizing MSE btw recon and

# loss_val = loss_fn(s * final_amp, target_amp)