
import torch
import torch.nn.functional as F
try:
    from lpips import LPIPS 
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Perceptual loss disabled.")

class JointLoss:
    def __init__(self, image_w=1.0, nav_w=0.1, vq_w=0.25, perceptual=False, device="cpu"):
        self.image_w = image_w
        self.nav_w = nav_w
        self.vq_w = vq_w
        self.perceptual = perceptual and LPIPS_AVAILABLE
        self.device = device
        if self.perceptual:
            self.lpips = LPIPS(net='vgg').to(device)

    def image_loss(self, recon, target):
        # Use Huber loss for more stability than L1
        l1 = F.smooth_l1_loss(recon, target, reduction='mean')
        if self.perceptual:
            p = self.lpips(recon, target).mean()
            return l1 + 0.01 * p  # Reduced perceptual weight
        return l1

    def nav_loss(self, pred_nav, gt_nav):
        # Use Huber loss for navigation too
        return F.smooth_l1_loss(pred_nav, gt_nav, reduction='mean')

    def __call__(self, recon, target, pred_nav, gt_nav, vq_loss):
        L_img = self.image_loss(recon, target)
        L_nav = self.nav_loss(pred_nav, gt_nav)
        
        # Ensure vq_loss is properly reduced for DataParallel
        if vq_loss.numel() > 1:
            vq_loss = vq_loss.mean()
        
        # Clamp VQ loss to prevent explosion
        vq_loss = torch.clamp(vq_loss, 0, 10.0)
            
        total = self.image_w * L_img + self.nav_w * L_nav + self.vq_w * vq_loss
        
        # Handle multi-GPU case where tensors might not be scalars for logging
        vq_loss_scalar = vq_loss.mean().item() if vq_loss.numel() > 1 else vq_loss.item()
        return total, {"L_img": L_img.item(), "L_nav": L_nav.item(), "vq": vq_loss_scalar}
