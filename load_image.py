import os, torch
# import pandas as pd
from torch.utils.data import Dataset, random_split
from torchvision.io import read_image
from PIL import Image
from torchvision import transforms

# NYU Depth Dataset V2
# https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

# A Large-Scale Hierarchical Multi-View RGB-D Object Dataset
# https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_full/
class LSHMV_RGBD_Object_Dataset(Dataset):
    def __init__(self, img_dir, color_transform=None, depth_transform=None, output_type='color_depth', channel=None):
        '''
        output_type can be chosen from 'color_depth', 'field', 'mask'
        '''
        # self.img_labels = pd.read_csv(annotations_file)
        # self.scene_list = ['scene_01', 'scene_02', 'scene_03', 'scene_04', 'scene_05', 'scene_06', 'scene_07', 
        #                    'scene_08', 'scene_09', 'scene_10', 'scene_11', 'scene_12', 'scene_13', 'scene_14']
        self.img_dir = img_dir
        img_list = os.listdir(img_dir)
        img_list.sort()
        self.color_list = [value for value in img_list if "color" in value]
        self.depth_list = [value for value in img_list if "depth" in value]
        self.color_transform = color_transform
        self.depth_transform = depth_transform
        self.channel = channel
        self.output_type = output_type

    def __len__(self):
        return len(self.color_list)

    def __getitem__(self, idx):
        color_path = os.path.join(self.img_dir, self.color_list[idx])
        depth_path = os.path.join(self.img_dir, self.depth_list[idx])
        # color_image = read_image(color_path)
        # depth_image = read_image(depth_path)
        color_image = Image.open(color_path)
        if self.channel:
            color_image = color_image.split()[self.channel]
        depth_image = Image.open(depth_path)
        
        if self.color_transform:
            color_image = self.color_transform(color_image)
        if self.depth_transform:
            depth_image = self.depth_transform(depth_image)
        
        # transform must contains ToTensor()
        if self.output_type == 'field':
            trans = transforms.ToTensor()
            if type(color_image) != torch.Tensor:
                color_image = trans(color_image)
            if type(depth_image) != torch.Tensor:
                depth_image = trans(depth_image)
            field = torch.cat([color_image, depth_image], 0)
            field = field.unsqueeze(0)
            return field
        elif self.output_type == 'color_depth':        
            return color_image, depth_image
        elif self.output_type == 'mask':
            pass
        else:
            raise RuntimeError("Undefined output_type, can only be chosen from 'color_depth', 'field', 'mask'")
    
    def load_img_mask():
        pass
        


if __name__ == '__main__':
    img_dir = '/home/wenbin/Downloads/rgbd-scenes-v2/imgs'
    scene_list = ['scene_01', 'scene_02', 'scene_03', 'scene_04', 'scene_05', 'scene_06', 'scene_07', 
                  'scene_08', 'scene_09', 'scene_10', 'scene_11', 'scene_12', 'scene_13', 'scene_14']
    nyu_dataset = LSHMV_RGBD_Object_Dataset('/home/wenbin/Downloads/rgbd-scenes-v2/imgs/scene_01', 
                                    #    channel=1 ,combine_output=True)
                                      )
    
    train_data_size = int(0.8*len(nyu_dataset))
    test_data_size = len(nyu_dataset)-train_data_size
    train_data, test_data = random_split(nyu_dataset, [train_data_size,test_data_size], generator=torch.Generator().manual_seed(42))
    
    # depth range around (5000~30000)mm
    
    pass