import torch
from load_image import LSHMV_RGBD_Object_Dataset
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from reverse3d_prop import Reverse3dProp
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import utils
from prop_model import CNNpropCNN_default


img_dir = '/home/wenbin/Downloads/rgbd-scenes-v2/imgs/scene_01'

tf = transforms.Compose([
    Resize((1080,1920)),
    ToTensor()
])
nyu_dataset = LSHMV_RGBD_Object_Dataset('/home/wenbin/Downloads/rgbd-scenes-v2/imgs/scene_01',
                                        color_transform=tf, depth_transform=tf,
                                        channel=1, output_type='mask')

for i in range(2, 15):
    scene_name = 'scene_' + str(i).zfill(2)
    nyu_dataset += LSHMV_RGBD_Object_Dataset('/home/wenbin/Downloads/rgbd-scenes-v2/imgs/'+scene_name,
                                             color_transform=tf, depth_transform=tf,
                                             channel=1, output_type='mask')
    
    
train_data_size = int(0.8*len(nyu_dataset))
test_data_size = len(nyu_dataset)-train_data_size
train_data, test_data = random_split(nyu_dataset, [train_data_size,test_data_size], generator=torch.Generator().manual_seed(42))

print(f"train set length: {train_data_size}")
print(f"test  set length: {test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=1)
test_dataloader = DataLoader(test_data, batch_size=1)

reverse_prop = Reverse3dProp()
reverse_prop = reverse_prop.cuda()
# for param in reverse_prop.CNNpropCNN.parameters():
#     param.requires_grad = False
forward_prop = CNNpropCNN_default()
forward_prop = forward_prop.cuda()
for param in forward_prop.parameters():
    param.requires_grad = False
loss_fn = nn.MSELoss().cuda()



learning_rate = 1e-2
optimizer = torch.optim.SGD(reverse_prop.parameters(), lr=learning_rate)

epoch = 100000

total_train_step = 0
best_test_loss = float('inf')
best_test_psnr = 0

# 添加tensorboard
time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
# writer = SummaryWriter("./logs")
writer = SummaryWriter(f'runs/{time_str}')

writer.add_scalar("learning_rate", learning_rate)

for i in range(epoch):
    print(f"----------Training Start (Epoch: {i+1})-------------")
    
    # training steps
    reverse_prop.train()
    for imgs_masks_id in train_dataloader:
        imgs, masks, imgs_id = imgs_masks_id
        imgs = imgs.cuda()
        masks = masks.cuda()
        # outputs_field = reverse_prop(imgs)
        # outputs_amp = outputs_field.abs()
        # final_amp = outputs_amp*masks
        slm_phase = reverse_prop(imgs)
        outputs_field = forward_prop(slm_phase)
        outputs_amp = outputs_field.abs()
        final_amp = outputs_amp*masks
        
        loss = loss_fn(final_amp, imgs)
        
        # a = utils.target_planes_to_one_image(final_amp, masks)
        # b = utils.target_planes_to_one_image(imgs, masks)
        
        with torch.no_grad(): 
            psnr = utils.calculate_psnr(utils.target_planes_to_one_image(final_amp, masks), utils.target_planes_to_one_image(imgs, masks))
        
        writer.add_scalar("train_loss", loss.item(), total_train_step)
        writer.add_scalar("train_psnr", psnr.item(), total_train_step)
        
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step = total_train_step + 1
        if (total_train_step) % 100 == 0:
            
            print(f"Training Step {total_train_step}, Loss: {loss.item()}")
            
            if (total_train_step) % 700 == 0:
                writer.add_text('image id', imgs_id[0], total_train_step)
                for i in range(8):
                    writer.add_image(f'input_images_plane{i}', imgs.squeeze()[i,:,:], total_train_step, dataformats='HW')
                    writer.add_images(f'output_image_plane{i}', outputs_amp.squeeze()[i,:,:], total_train_step, dataformats='HW')
        
    # test the model after every epoch
    reverse_prop.eval()
    total_test_loss = 0
    total_test_psnr = 0
    test_items_count = 0
    with torch.no_grad():
        for imgs_masks_id in test_dataloader:
            imgs, masks, imgs_id = imgs_masks_id
            imgs = imgs.cuda()
            masks = masks.cuda()
            # outputs_field = reverse_prop(imgs)
            # outputs_amp = outputs_field.abs()
            # final_amp = outputs_amp * masks
            
            slm_phase = reverse_prop(imgs)
            outputs_field = forward_prop(slm_phase)
            outputs_amp = outputs_field.abs()
            final_amp = outputs_amp*masks
            
            # outputs = reverse_prop(imgs)
            loss = loss_fn(final_amp, imgs)
            psnr = utils.calculate_psnr(utils.target_planes_to_one_image(final_amp, masks), utils.target_planes_to_one_image(imgs, masks))
            
            total_test_loss += loss
            total_test_psnr += psnr
            test_items_count += 1
        
        average_test_loss = total_test_loss/test_items_count
        average_test_psnr = total_test_psnr/test_items_count
        if best_test_loss > average_test_loss:
            best_test_loss = average_test_loss
            # save model
            path = f"runs/{time_str}/model/"
            if not os.path.exists(path):
                os.makedirs(path) 
            torch.save(reverse_prop, f"runs/{time_str}/model/reverse_3d_prop_{time_str}_best_loss.pth")
            print("model saved!")
            
    print(f"Average Test Loss: {average_test_loss}")
    print(f"Average Test PSNR: {average_test_psnr}")
    writer.add_scalar("average_test_loss", average_test_loss.item(), total_train_step)
    writer.add_scalar("average_test_psnr", average_test_psnr.item(), total_train_step)
    
    

# with torch.no_grad():
#     s = (final_amp * target_amp).mean() / \
#         (final_amp ** 2).mean()  # scale minimizing MSE btw recon and

# loss_val = loss_fn(s * final_amp, target_amp)