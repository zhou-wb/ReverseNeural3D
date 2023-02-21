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
        
        trans = transforms.ToTensor()
        if type(color_image) != torch.Tensor:
            color_image = trans(color_image)
        if type(depth_image) != torch.Tensor:
            depth_image = trans(depth_image)
        
        # transform must contains ToTensor()
        if self.output_type == 'field':            
            field = torch.cat([color_image, depth_image], 0)
            # field = field.unsqueeze(0)
            return field
        elif self.output_type == 'color_depth':        
            return color_image, depth_image
        elif self.output_type == 'mask':
            depth_image = self.depth_convert(depth_image)
            masks = self.load_img_mask(depth_image)
            return (color_image*masks).squeeze(), masks, self.img_dir.split('/')[-1]+'-'+self.color_list[idx]
        else:
            raise RuntimeError("Undefined output_type, can only be chosen from 'color_depth', 'field', 'mask'")
    
    def depth_convert(self, depth):
        # NaN to inf
        depth[depth==0] = 100000
        # convert to double
        depth = depth.double()
        # convert mm to m
        depth = depth/1000
        # this gives us decent depth distribution with 120mm eyepiece setting.
        depth /= 2.5
        # meter to diopter conversion
        depth = 1 / (depth + 1e-20)
        
        # depth = depth.unsqueeze(0)
        return depth
    
    def load_img_mask(self, depth):               
        """ decompose a depthmap image into a set of masks with depth positions (in Diopter) """

        self.virtual_depth_planes = [0.0, 0.08417508417508479, 0.14124293785310726, 0.24299599771297942, 0.3171856978085348, 0.4155730533683304, 0.5319148936170226, 0.6112104949314254]
        
        depth_planes_D = self.virtual_depth_planes
        depthmap_virtual_D = depth
        
        num_planes = len(depth_planes_D)

        masks = torch.zeros(depthmap_virtual_D.shape[0], len(depth_planes_D), *depthmap_virtual_D.shape[-2:],
                            dtype=torch.float32).to(depthmap_virtual_D.device)
        for k in range(len(depth_planes_D) - 1):
            depth_l = depth_planes_D[k]
            depth_h = depth_planes_D[k + 1]
            idxs = (depthmap_virtual_D >= depth_l) & (depthmap_virtual_D < depth_h)
            close_idxs = (depth_h - depthmap_virtual_D) > (depthmap_virtual_D - depth_l)

            # closer one
            mask = torch.zeros_like(depthmap_virtual_D)
            mask += idxs * close_idxs * 1
            masks[:, k, ...] += mask.squeeze(1)

            # farther one
            mask = torch.zeros_like(depthmap_virtual_D)
            mask += idxs * (~close_idxs) * 1
            masks[:, k + 1, ...] += mask.squeeze(1)

        # even closer ones
        idxs = depthmap_virtual_D >= max(depth_planes_D)
        mask = torch.zeros_like(depthmap_virtual_D)
        mask += idxs * 1
        masks[:, len(depth_planes_D) - 1, ...] += mask.clone().squeeze(1)

        # even farther ones
        idxs = depthmap_virtual_D < min(depth_planes_D)
        mask = torch.zeros_like(depthmap_virtual_D)
        mask += idxs * 1
        masks[:, 0, ...] += mask.clone().squeeze(1)

        # sanity check
        assert torch.sum(masks).item() == torch.numel(masks) / num_planes

        return masks.squeeze()
        


if __name__ == '__main__':
    img_dir = '/home/wenbin/Downloads/rgbd-scenes-v2/imgs'
    scene_list = ['scene_01', 'scene_02', 'scene_03', 'scene_04', 'scene_05', 'scene_06', 'scene_07', 
                  'scene_08', 'scene_09', 'scene_10', 'scene_11', 'scene_12', 'scene_13', 'scene_14']
    
    tf = transforms.Compose([
        transforms.Resize((1080,1920)),
        transforms.ToTensor()
    ])
    
    nyu_dataset = LSHMV_RGBD_Object_Dataset('/home/wenbin/Downloads/rgbd-scenes-v2/imgs/scene_01', 
                                       channel=1 ,output_type='mask', color_transform=tf, depth_transform=tf)
                                    #   )
    
    train_data_size = int(0.8*len(nyu_dataset))
    test_data_size = len(nyu_dataset)-train_data_size
    train_data, test_data = random_split(nyu_dataset, [train_data_size,test_data_size], generator=torch.Generator().manual_seed(42))
    
    # depth range around (5000~30000)mm
    
    print(train_data[0])
    
    
    pass