import torch
import torch.nn.functional as F
# from torchmetrics.image import TotalVariation
from torchmetrics.functional.image import total_variation

class MSETVLoss(torch.nn.Module):
    def __init__(self):
        super(MSETVLoss, self).__init__()
        # self.tv = TotalVariation()
    
    def forward(self, recon_amp, target_amp, phase):
        return F.mse_loss(recon_amp, target_amp, reduction='mean') + 2e-10 * total_variation(phase)