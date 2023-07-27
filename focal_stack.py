import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm

from load_flying3d import FlyingThings3D_loader

def get_kernel2d(kernel_size, valid_ratio, focus=0., normalize=False, device=torch.device('cpu')):
    """为局部区域的加权求和生成2d的核权重。

    Args:
        kernel_size (int): 核的边长。
        valid_ratio (torch.Tensor): 有效区间比例（基于边长的一半）。
        normalize (bool, optional): 是否归一化输出。

    Returns:
        torch.Tensor: 生成的核 (kernel_size*kernel_size, H, W)
    """
    assert valid_ratio.min() >= 0 and valid_ratio.max() <= 1
    valid_ratio = torch.abs(valid_ratio - focus)
    ori_shape = valid_ratio.shape
    valid_ratio = -torch.log(1-valid_ratio)

    valid_width = kernel_size // 2 * valid_ratio
    coords = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size).reshape(-1, 1, 1).to(device)

    kernel_1d = F.relu(1 - torch.abs(coords / valid_width)).nan_to_num(nan=1)
    kernel_2d = torch.einsum("xhw,yhw->xyhw", kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.reshape(-1, *ori_shape)

    if normalize:
        kernel_2d = kernel_2d / kernel_2d.sum(dim=0)
    return kernel_2d

def render_focal_stack(image, mask, total_num_of_planes, plane_ids, kernel_size, device=torch.device('cpu')):
    """render focal stack for rgbd input

    Args:
        image (torch.Tensor): rgb image get from data loader
        mask (torch.Tensor): mask get from data loader
        total_num_of_planes: Number of planes which the space is divided into
        plane_ids: subset of [0,1,2,...,total_num_of_planes-1], its length should be the same as the number of channels in mask
        focus_plane_dists (list of float): specify all the focus plane distances

    Returns:
        torch.Tensor: rendered focal stack, len(focus_plane_dists) * image.width * image.height
    """
    assert len(mask) == len(plane_ids)
    
    dists = [idx * (1./(total_num_of_planes-1)) for idx in plane_ids]
    fake_depth = torch.sum(mask*torch.tensor(dists)[:,None,None].to(device), dim = 0)
    
    num_channels, height, width  = image.shape


    tgt_image = torch.zeros_like(image, dtype=torch.float32, device=device)

    # image = image.permute(2, 0, 1).unsqueeze(0)
    image = image.unsqueeze(0)
    image = F.pad(image, pad=(kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='replicate')
    image = F.unfold(image, kernel_size=kernel_size)
    image = image.reshape(num_channels, kernel_size*kernel_size, height, width)

    focal_stack = []
    for idx, dist in enumerate(dists):
        weight = get_kernel2d(kernel_size=kernel_size, valid_ratio=fake_depth, focus=dist, normalize=True, device=device)

        tgt_image = (image * weight).sum(dim=1)
        focal_stack.append(tgt_image)
        # tgt_image = tgt_image.permute(1, 2, 0)
    
    focal_stack = torch.stack(focal_stack)
    return focal_stack
    # cv2.imwrite(f'./result/blur{idx}.png', (tgt_image*255).astype(np.uint8))

if __name__ == '__main__':
    data_path = '/media/datadrive/flying3D'
    output_root = '/media/datadrive/focal-stack/flying3D'
    
    image_res = (540, 960)
    roi_res = (540, 960)
    virtual_depth_planes = [0.0, 0.08417508417508479, 0.14124293785310726, 0.24299599771297942, 0.3171856978085348, 0.4155730533683304, 0.5319148936170226, 0.6112104949314254]
    random_seed = 10
    device = torch.device('cuda:0')
    train_loader = FlyingThings3D_loader(data_path=data_path,
                                         channel=1, image_res=image_res, roi_res=roi_res,
                                         virtual_depth_planes=virtual_depth_planes,
                                         return_type='image_mask_id',
                                         random_seed=random_seed,
                                         # slice=(0,0.025),
                                         # slice=(0,0.1),
                                         slice=(0,0.025),
                                         scale_vd_range=True,
                                        )
    val_loader = FlyingThings3D_loader(data_path=data_path,
                                       channel=1, image_res=image_res, roi_res=roi_res,
                                       virtual_depth_planes=virtual_depth_planes,
                                       return_type='image_mask_id',
                                       random_seed=random_seed,
                                       # slice=(0.995,1),
                                       slice=(0.2,0.205),
                                       )
    
    for image, mask, path in tqdm(val_loader):
        image = image.to(device)
        mask = mask.to(device)
        focal_stack = render_focal_stack(image, mask, total_num_of_planes=8, plane_ids=list(range(8)), kernel_size=21, device=device)
        output_folder = os.path.join(output_root, path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        torch.save(focal_stack, os.path.join(output_folder, 'focalstack.pth'))
        for i in range(len(virtual_depth_planes)):
            cv2.imwrite(os.path.join(output_folder, f'focalstack{i}.png'), (focal_stack[i].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
        
    
    
    
    
    