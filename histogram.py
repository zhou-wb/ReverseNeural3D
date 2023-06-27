import matplotlib.pyplot as plt
from load_image import LSHMV_RGBD_Object_Dataset
from torch.utils.data import Subset
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
import numpy as np
from tqdm import tqdm

def compute_histogram(tensor, bins, range=None):
    # Convert tensor to numpy array
    np_array = tensor.numpy()
    # Compute histogram
    hist, bin_edges = np.histogram(np_array, bins=bins, range=range)
    # remove 0
    # hist[0] -= np.count_nonzero(np_array==0) 
    return hist, bin_edges

def depth_convert(depth):
    depth[depth==0] = 100000
    depth = 1/depth
    
    value, _ = torch.sort(torch.flatten(depth))
    top10_value = value[int(-len(value)*0.008)]
    top1_value = value[-1]
    resize_factor = top10_value / 0.61
    depth = depth / resize_factor
    
    return depth

img_dir = '/home/wenbin/Downloads/rgbd-scenes-v2/imgs/scene_01'

tf = transforms.Compose([
    Resize((1080,1920)),
    ToTensor()
])

nyu_dataset = LSHMV_RGBD_Object_Dataset('/home/wenbin/Downloads/rgbd-scenes-v2/imgs/scene_01',
                                        color_transform=tf, depth_transform=tf,
                                        channel=1, output_type='color_depth')

for i in range(2, 15):
    scene_name = 'scene_' + str(i).zfill(2)
    nyu_dataset += LSHMV_RGBD_Object_Dataset('/home/wenbin/Downloads/rgbd-scenes-v2/imgs/'+scene_name,
                                             color_transform=tf, depth_transform=tf,
                                             channel=1, output_type='color_depth')

num_of_planes = 8
sum_hist = [0]*num_of_planes
n_samples = 500
indices = np.random.choice(len(nyu_dataset), n_samples, replace=False)
for rgb_img, depth_img in tqdm(Subset(nyu_dataset, indices)):
# for rgb_img, depth_img in tqdm(nyu_dataset):
    # depth_img = depth_convert(depth_img)
    # hist, bin_edges = compute_histogram(depth_img, num_of_planes, range=(0, 0.61))
    hist, bin_edges = compute_histogram(depth_img, num_of_planes)
    sum_hist = [a + b for a, b in zip(sum_hist, hist)]

# hist, bin_edges = compute_histogram(nyu_dataset[0][1], num_of_planes, range=(0, 30000))

# Plot histogram
width = bin_edges[1] - bin_edges[0]
plt.bar(bin_edges[:-1], sum_hist, width=width, align='edge')
plt.xlim(min(bin_edges), max(bin_edges))
plt.savefig('histogram.png')

    
    
# for imgs_masks_id in train_dataloader:
#         imgs, masks, imgs_id = imgs_masks_id

# plt.hist(x, bins=number of bins)
# plt.show()