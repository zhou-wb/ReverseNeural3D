import torch
from load_image import LSHMV_RGBD_Object_Dataset
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize

img_dir = '/home/wenbin/Downloads/rgbd-scenes-v2/imgs/scene_01'

tf = transforms.Compose([
    Resize((1080,1920)),
    ToTensor()
])
nyu_dataset = LSHMV_RGBD_Object_Dataset('/home/wenbin/Downloads/rgbd-scenes-v2/imgs/scene_01',
                                   color_transform=tf, depth_transform=tf,
                                   channel=1, output_type=True)
    
train_data_size = int(0.8*len(nyu_dataset))
test_data_size = len(nyu_dataset)-train_data_size
train_data, test_data = random_split(nyu_dataset, [train_data_size,test_data_size], generator=torch.Generator().manual_seed(42))

print(f"train set length: {train_data_size}")
print(f"test  set length: {test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=32)
test_dataloader = DataLoader(test_data, batch_size=32)

