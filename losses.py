
import torch
import torch.nn.functional as F
from lpips import LPIPS 

class JointLoss:
    def __init__(self, image_w=1.0, nav_w=1.0, perceptual=False, device="cpu"):
        self.image_w = image_w
        self.nav_w = nav_w
        self.perceptual = perceptual
        self.device = device
        if perceptual:
            self.lpips = LPIPS(net='vgg').to(device)

    def image_loss(self, recon, target):
        l1 = F.l1_loss(recon, target, reduction='mean')
        if self.perceptual:
            p = self.lpips(recon, target).mean()
            return l1 + 0.1 * p
        return l1

    def nav_loss(self, pred_nav, gt_nav):
        return F.mse_loss(pred_nav, gt_nav, reduction='mean')

    def __call__(self, recon, target, pred_nav, gt_nav, vq_loss):
        L_img = self.image_loss(recon, target)
        L_nav = self.nav_loss(pred_nav, gt_nav)
        
        # Ensure vq_loss is properly reduced for DataParallel
        if vq_loss.numel() > 1:
            vq_loss = vq_loss.mean()
            
        total = self.image_w * L_img + self.nav_w * L_nav + vq_loss
        
        # Handle multi-GPU case where tensors might not be scalars for logging
        vq_loss_scalar = vq_loss.mean().item() if vq_loss.numel() > 1 else vq_loss.item()
        return total, {"L_img": L_img.item(), "L_nav": L_nav.item(), "vq": vq_loss_scalar}
