import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
from dataset import make_dataloader
from models import DronePredictor
from losses import JointLoss
from utils import save_checkpoint
from config import cfg

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_worker(rank, world_size):
    """Training function for each GPU process."""
    setup(rank, world_size)
    
    device = f'cuda:{rank}'
    
    # Create dataset with distributed sampler
    from dataset import DroneSeqDataset
    from torch.utils.data import DataLoader
    
    dataset = DroneSeqDataset(cfg.data_root, seq_len=cfg.seq_len, annotation_dir=cfg.annotation_root)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, 
                       num_workers=cfg.num_workers//world_size, pin_memory=True)
    
    # Create model and wrap with DDP
    model = DronePredictor(cfg).to(device)
    model = DDP(model, device_ids=[rank])
    
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = JointLoss(image_w=cfg.image_loss_weight, nav_w=cfg.nav_loss_weight, 
                         perceptual=False, device=device)

    global_step = 0
    for epoch in range(cfg.epochs):
        # Set epoch for distributed sampler
        sampler.set_epoch(epoch)
        
        model.train()
        if rank == 0:  # Only show progress bar on main process
            pbar = tqdm(loader, desc=f"Epoch {epoch}")
        else:
            pbar = loader
            
        running_loss = 0.0
        for batch in pbar:
            imgs = batch['imgs'].to(device, non_blocking=True)
            navs = batch['navs'].to(device, non_blocking=True)
            tgt_img = batch['tgt_img'].to(device, non_blocking=True)
            tgt_nav = batch['tgt_nav'].to(device, non_blocking=True)

            opt.zero_grad()
            recon, nav_pred, vq_loss = model(imgs, navs)
            loss, info = criterion(recon, tgt_img, nav_pred, tgt_nav, vq_loss)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if rank == 0:  # Only update progress bar on main process
                pbar.set_description(f"Epoch {epoch} loss {running_loss/(global_step+1):.4f}")
            global_step += 1

        # Save checkpoint only on main process
        if rank == 0:
            ckpt = {
                "model": model.module.state_dict(),  # Remove DDP wrapper
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "cfg": vars(cfg)
            }
            save_checkpoint(ckpt, os.path.join(cfg.checkpoint_dir, f"ckpt_epoch{epoch}.pt"))
            print(f"Epoch {epoch} completed. Checkpoint saved.")

    cleanup()

def train_distributed():
    """Main function to launch distributed training."""
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Distributed training requires at least 2 GPUs. Falling back to single GPU training.")
        return False
    
    print(f"Starting distributed training on {world_size} GPUs")
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)
    return True

if __name__ == "__main__":
    if not train_distributed():
        # Fallback to regular training
        from train import train
        train()