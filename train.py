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


img_dir = '/home/wenbin/Downloads/rgbd-scenes-v2/imgs/scene_01'

tf = transforms.Compose([
    Resize((1080,1920)),
    ToTensor()
])
nyu_dataset = LSHMV_RGBD_Object_Dataset('/home/wenbin/Downloads/rgbd-scenes-v2/imgs/scene_01',
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
loss_fn = nn.MSELoss().cuda()

learning_rate = 1e-2
optimizer = torch.optim.SGD(reverse_prop.parameters(), lr=learning_rate)

epoch = 10

total_train_step = 0

# 添加tensorboard
time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
# writer = SummaryWriter("./logs")
writer = SummaryWriter(f'runs/{time_str}')

for i in range(epoch):
    print(f"----------第 {i+1} 轮训练开始----------")
    
    # 训练步骤开始
    reverse_prop.train()
    for imgs in train_dataloader:
        imgs = imgs.cuda()
        outputs_field = reverse_prop(imgs)
        outputs_amp = outputs_field.abs()
        loss = loss_fn(outputs_amp, imgs)
        
        writer.add_scalar("train_loss", loss.item(), total_train_step)
        
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step = total_train_step + 1
        if (total_train_step) % 100 == 0:
            
            print(f"训练次数 {total_train_step}, Loss: {loss.item()}")
            
            for i in range(8):
                writer.add_image(f'input_images_plane{i}', imgs.squeeze()[i,:,:], total_train_step, dataformats='HW')
                writer.add_images(f'output_image_plane{i}', outputs_amp.squeeze()[i,:,:], total_train_step, dataformats='HW')
        
    # 测试步骤
    reverse_prop.eval()
    total_test_loss = 0
    with torch.no_grad():
        for imgs in test_dataloader:
            imgs = imgs.cuda()
            outputs_field = reverse_prop(imgs)
            outputs_amp = outputs_field.abs()
            # outputs = reverse_prop(imgs)
            loss = loss_fn(outputs_amp, imgs)
            total_test_loss += loss
            
            
    print(f"整体测试集上的Loss: {total_test_loss}")
    writer.add_scalar("test_loss", total_test_loss.item(), total_train_step)
    
    
    # 保存模型文件
    # path = f"runs/{time_str}/model/"
    # if not os.path.exists(path):
    #     os.makedirs(path) 
    # torch.save(reverse_prop, f"runs/{time_str}/model/reverse_3d_prop_{time_str}_{i}.pth")
    # print("模型已保存")


# with torch.no_grad():
#     s = (final_amp * target_amp).mean() / \
#         (final_amp ** 2).mean()  # scale minimizing MSE btw recon and

# loss_val = loss_fn(s * final_amp, target_amp)