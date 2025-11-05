'''
The encoder reduces spatial size twice (Conv strides 2) — final h,w depend on input size and chosen transforms.
In the dataset.py I resized to (256,448) to make h,w manageable.
For full-res training we will likely want a much larger encoder and more compute.

We can replace the SimpleTransformer and VectorQuantizer with more sophisticated modules (Perceiver, TimeSformer, VPTR variants)
depending on our compute budget and experiments evolve. See citations.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# conv encoder/decoder + VQ 
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
            nn.Sigmoid()
        )
    def forward(self,z): return self.net(z)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight,-1/num_embeddings,1/num_embeddings)
        self.commitment_cost = commitment_cost
    def forward(self, z):
        # z: (B, C, H, W)
        B, C, H, W = z.shape
        flat = z.permute(0,2,3,1).contiguous().view(-1, C)  # (B*H*W, C)
        # compute distances
        d = torch.sum(flat**2, dim=1, keepdim=True) - 2*flat @ self.embedding.weight.t() + torch.sum(self.embedding.weight**2, dim=1)
        indices = torch.argmin(d, dim=1)
        quantized = self.embedding(indices).view(B, H, W, C).permute(0,3,1,2).contiguous()
        # losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
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

# full model 
class DronePredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # encoder/decoder/vq
        self.encoder = ConvEncoder(in_ch=3, hidden=256, z_channels=cfg.d_model//8)  # tune
        z_ch = cfg.d_model//8
        self.vq = VectorQuantizer(num_embeddings=cfg.vq_codebook_size, embedding_dim=z_ch)
        self.decoder = ConvDecoder(z_channels=z_ch, hidden=256, out_ch=3)

        # transformer predicts flattened token embeddings
        # token embedding: use codebook vectors directly (we embed indices to d_model)
        self.codebook_proj = nn.Linear(z_ch, cfg.d_model)
        self.codebook_unproj = nn.Linear(cfg.d_model, z_ch)

        self.transformer = SimpleTransformer(d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers, dropout=cfg.dropout)
        # temporal positional embeddings for transformer
        # Increase max sequence length to handle large spatial resolutions
        max_seq_len = cfg.seq_len * cfg.latent_h * cfg.latent_w * 4  # Extra buffer for safety
        self.temporal_pos = nn.Parameter(torch.randn(max_seq_len, cfg.d_model))
        # navigation regression head
        self.nav_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model//2), nn.GELU(),
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
