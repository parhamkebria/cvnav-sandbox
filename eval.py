import torch
from dataset import make_dataloader
from models import DronePredictor
from config import cfg
from utils import load_checkpoint
from torchvision.utils import save_image
import os

def evaluate(ckpt_path, out_dir="./eval_out"):
    device = cfg.device
    loader = make_dataloader(cfg.data_root, seq_len=cfg.seq_len, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    model = DronePredictor(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck['model'])
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch['imgs'].to(device)
            navs = batch['navs'].to(device)
            tgt_img = batch['tgt_img'].to(device)
            tgt_nav = batch['tgt_nav'].to(device)
            recon, nav_pred, _ = model(imgs, navs)
            # save images
            save_image(recon.clamp(0,1), os.path.join(out_dir, f"pred_{i:05d}.png"))
            save_image(tgt_img, os.path.join(out_dir, f"gt_{i:05d}.png"))
            # save navs
            with open(os.path.join(out_dir, f"nav_{i:05d}.txt"), "w") as f:
                f.write("pred: " + ", ".join([f"{v:.6f}" for v in nav_pred.squeeze().cpu().numpy()]) + "\n")
                f.write("gt:   " + ", ".join([f"{v:.6f}" for v in tgt_nav.squeeze().cpu().numpy()]) + "\n")
            if i > 200:
                break

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1]
    evaluate(ckpt)
