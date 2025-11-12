'''
Enhanced encoder-decoder with VGG-16 backbone for better feature extraction.
Uses VGG-16 conv5 features (14x14x512).
Optimized for better GPU utilization and performance.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as models

# VGG-16 based encoder
class VGGEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        # Load VGG-16 and use features up to conv5
        vgg16 = models.vgg16(pretrained=pretrained)
        
        # Extract features up to conv5 (index 30 in VGG-16 features)
        # This gives us 7x7x512 feature maps for 224x224 input
        self.backbone = nn.Sequential(*list(vgg16.features.children())[:31])  # Up to conv5_3
        
        # Optionally freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Additional conv to reduce to desired latent dimension
        self.latent_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, 1, 1),  # Final latent dimension
        )
        
    def forward(self, x):
        # x: (B, 3, 224, 224)
        features = self.backbone(x)  # (B, 512, 7, 7)
        latent = self.latent_conv(features)  # (B, 64, 7, 7)
        return latent

# Enhanced decoder for VGG features
class VGGDecoder(nn.Module):
    def __init__(self, z_channels=64, out_ch=3):
        super().__init__()
        # Upsample from 7x7 back to 224x224
        self.net = nn.Sequential(
            nn.Conv2d(z_channels, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            
            # Upsample to 14x14
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            
            # Upsample to 28x28
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            
            # Upsample to 56x56
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            
            # Upsample to 112x112
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            
            # Upsample to 224x224
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            
            # Final output layer
            nn.Conv2d(16, out_ch, 3, 1, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, z): 
        # Get tanh output in [-1, 1]
        tanh_output = self.net(z)
        # Convert tanh output to [0, 1] range first
        normalized_01 = (tanh_output + 1.0) / 2.0
        # Then apply ImageNet normalization: (x - mean) / std
        # Use hardcoded values to avoid register_buffer issues with DataParallel
        mean = torch.tensor([0.485, 0.456, 0.406], device=z.device, dtype=z.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=z.device, dtype=z.dtype).view(1, 3, 1, 1)
        return (normalized_01 - mean) / std

# Keep the original simple encoder/decoder for comparison
class ConvEncoder(nn.Module):
    def __init__(self, in_ch=3, hidden=128, z_channels=64):
        super().__init__()
        # input (B,C,H,W)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden//2, 4, 2, 1), nn.GELU(),
            nn.Conv2d(hidden//2, hidden, 4, 2, 1), nn.GELU(),
            nn.Conv2d(hidden, z_channels, 3, 1, 1)
        )
    def forward(self,x): return self.net(x)

class ConvDecoder(nn.Module):
    def __init__(self, z_channels=64, hidden=128, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_channels, hidden, 3,1,1), nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden//2, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(hidden//2, out_ch, 4, 2, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, z): 
        # Get tanh output in [-1, 1]
        tanh_output = self.net(z)
        # Convert tanh output to [0, 1] range first
        normalized_01 = (tanh_output + 1.0) / 2.0
        # Then apply ImageNet normalization: (x - mean) / std
        # Use hardcoded values to avoid register_buffer issues with DataParallel
        mean = torch.tensor([0.485, 0.456, 0.406], device=z.device, dtype=z.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=z.device, dtype=z.dtype).view(1, 3, 1, 1)
        return (normalized_01 - mean) / std

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Better initialization for stability
        nn.init.normal_(self.embedding.weight, 0, 0.02)
        self.commitment_cost = commitment_cost
        
    def forward(self, z):
        # z: (B, C, H, W)
        B, C, H, W = z.shape
        flat = z.permute(0,2,3,1).contiguous().view(-1, C)  # (B*H*W, C)
        
        # L2 normalize inputs for more stable quantization
        flat_norm = F.normalize(flat, p=2, dim=1)
        embedding_norm = F.normalize(self.embedding.weight, p=2, dim=1)
        
        # compute distances
        d = torch.sum(flat_norm**2, dim=1, keepdim=True) - 2*flat_norm @ embedding_norm.t() + torch.sum(embedding_norm**2, dim=1)
        indices = torch.argmin(d, dim=1)
        quantized = self.embedding(indices).view(B, H, W, C).permute(0,3,1,2).contiguous()
        
        # losses with gradient clipping
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Clamp VQ loss for stability
        loss = torch.clamp(loss, 0, 5.0)
        
        quantized = z + (quantized - z).detach()
        indices = indices.view(B, H, W)
        return quantized, indices, loss

# transformer predictor 
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4, dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_emb = None
    def forward(self, x, src_key_padding_mask=None):
        # x: (B, S, D) batch-first for better performance
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)

# Enhanced model with VGG-16 backbone
class DronePredictor(nn.Module):
    def __init__(self, cfg, use_vgg=True):
        super().__init__()
        
        # Choose encoder type
        if use_vgg:
            self.encoder = VGGEncoder(pretrained=True, freeze_backbone=False)
            z_ch = 64  # VGG encoder output channels
        else:
            self.encoder = ConvEncoder(in_ch=3, hidden=256, z_channels=cfg.d_model//8)
            z_ch = cfg.d_model//8
            
        self.vq = VectorQuantizer(num_embeddings=cfg.vq_codebook_size, embedding_dim=z_ch)
        
        # Choose decoder type
        if use_vgg:
            self.decoder = VGGDecoder(z_channels=z_ch, out_ch=3)
        else:
            self.decoder = ConvDecoder(z_channels=z_ch, hidden=256, out_ch=3)

        # transformer predicts flattened token embeddings
        self.codebook_proj = nn.Linear(z_ch, cfg.d_model)
        self.codebook_unproj = nn.Linear(cfg.d_model, z_ch)

        self.transformer = SimpleTransformer(d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers, dropout=cfg.dropout)
        
        # Dynamic positional embeddings - compute based on actual feature map size
        self.max_spatial_tokens = 2048  # Reasonable upper bound for spatial tokens
        max_seq_len = cfg.seq_len * self.max_spatial_tokens
        self.temporal_pos = nn.Parameter(torch.randn(max_seq_len, cfg.d_model))
        
        # navigation regression head
        self.nav_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model//2), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model//2, 6)  # lat,lon,alt,roll,pitch,yaw
        )

    def encode_imgs(self, imgs):
        # imgs: (B, seq_len, C, H, W)
        B, S, C, H, W = imgs.shape
        imgs = imgs.view(B*S, C, H, W)
        z = self.encoder(imgs)                    # (B*S, zC, h, w)
        quantized, indices, vq_loss = self.vq(z)
        # convert quantized to tokens: flatten grid positions
        BSp, zC, h, w = quantized.shape
        # project to d_model token embeddings per spatial location
        proj = quantized.permute(0,2,3,1).contiguous().view(B, S, h*w, zC)  # (B,S,N,zC)
        proj = self.codebook_proj(proj)  # (B,S,N,d_model)
        return proj, indices, vq_loss, (h,w)

    def decode_tokens(self, token_embeddings, spatial_shape):
        # token_embeddings: (B, N, d_model) where N = h*w
        B, N, D = token_embeddings.shape
        h,w = spatial_shape
        z = self.codebook_unproj(token_embeddings)  # (B,N,zC)
        z = z.view(B, h, w, -1).permute(0,3,1,2).contiguous()
        recon = self.decoder(z)
        return recon

    def forward(self, imgs, navs):
        # imgs: (B, seq_len, C, H, W), navs: (B, seq_len, 6)
        B, S, C, H, W = imgs.shape
        proj, indices, vq_loss, spatial = self.encode_imgs(imgs)
        # flatten time+space into sequence
        B,S,N,D = proj.shape
        seq = proj.view(B, S*N, D)        # (B, L, D)
        # add positional/temporal embeddings
        # create pos emb of length L
        L = seq.size(1)
        # Handle cases where L exceeds pre-allocated positional embeddings
        if L > self.temporal_pos.size(0):
            # Extend positional embeddings if needed
            additional_pos = torch.randn(L - self.temporal_pos.size(0), self.temporal_pos.size(1), 
                                        device=self.temporal_pos.device, dtype=self.temporal_pos.dtype)
            pos_emb = torch.cat([self.temporal_pos, additional_pos], dim=0)
        else:
            pos_emb = self.temporal_pos
        
        pos = pos_emb[:L].unsqueeze(0).expand(B, -1, -1)  # (B,L,D)
        seq_t = seq + pos  # (B,L,D)
        out = self.transformer(seq_t)     # (B,L,D)
        # we want to predict the NEXT frame's tokens. 
        # simple scheme: take the final token outputs corresponding to last time-step forecast anchor:
        # pick slice corresponding to last time step's positions (last N tokens)
        last_out = out[:, -N:, :]         # (B,N,D)
        # decode image
        recon = self.decode_tokens(last_out, spatial)
        # nav regression using mean pooling of transformer's last time outputs
        pooled = out.mean(dim=1)          # (B,D)
        nav_pred = self.nav_head(pooled)  # (B,6)
        return recon, nav_pred, vq_loss

def get_model(cfg, use_vgg=True):
    """Factory function to create the model with optional VGG backbone"""
    return DronePredictor(cfg, use_vgg=use_vgg)
