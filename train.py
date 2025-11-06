import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
from dataset import make_dataloader
from models import get_model
from losses import JointLoss
from utils import save_checkpoint
from config import cfg
import csv
from datetime import datetime
from utils import save_nav_stats, print_nav_stats_summary

import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

def validate(model, val_loader, criterion, device):
    """Run validation and return average losses"""
    model.eval()
    total_loss = 0.0
    total_img_loss = 0.0
    total_nav_loss = 0.0
    total_vq_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            imgs = batch['imgs'].to(device)
            navs = batch['navs'].to(device)
            tgt_img = batch['tgt_img'].to(device)
            tgt_nav = batch['tgt_nav'].to(device)
            
            recon, nav_pred, vq_loss = model(imgs, navs)
            loss, info = criterion(recon, tgt_img, nav_pred, tgt_nav, vq_loss)
            
            # Handle multi-GPU case
            loss_scalar = loss.mean().item() if loss.numel() > 1 else loss.item()
            total_loss += loss_scalar
            total_img_loss += info['L_img']
            total_nav_loss += info['L_nav']
            total_vq_loss += info['vq']
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'img_loss': total_img_loss / num_batches,
        'nav_loss': total_nav_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches
    }

def setup_csv_loggers():
    """Setup CSV loggers with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    train_csv_path = f"logs/training_loss_{timestamp}.csv"
    val_csv_path = f"logs/val_loss_{timestamp}.csv"
    
    # Initialize training CSV
    with open(train_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'iteration', 'global_step', 'total_loss', 'img_loss', 
            'nav_loss', 'vq_loss', 'learning_rate', 'timestamp'
        ])
    
    # Initialize validation CSV
    with open(val_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'total_loss', 'img_loss', 'nav_loss', 'vq_loss', 'timestamp'
        ])
    
    return train_csv_path, val_csv_path

def log_training_iteration(csv_path, epoch, iteration, global_step, loss_scalar, info, lr):
    """Log training iteration to CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, iteration, global_step, loss_scalar, info['L_img'], 
            info['L_nav'], info['vq'], lr, timestamp
        ])

def log_validation_epoch(csv_path, epoch, val_metrics):
    """Log validation epoch to CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, val_metrics['total_loss'], val_metrics['img_loss'], 
            val_metrics['nav_loss'], val_metrics['vq_loss'], timestamp
        ])

def train():
    device = cfg.device
    
    # Check for multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        # Increase batch size proportionally to number of GPUs
        effective_batch_size = cfg.batch_size * torch.cuda.device_count()
        print(f"Effective batch size: {effective_batch_size}")
    else:
        effective_batch_size = cfg.batch_size
    
    # Create dataset with proper normalization
    from dataset import DroneSeqDataset
    from torch.utils.data import DataLoader, random_split, Subset
    
    print("Creating dataset with normalization...")
    full_dataset = DroneSeqDataset(cfg.data_root, seq_len=cfg.seq_len, annotation_dir=cfg.annotation_root)
    nav_stats = full_dataset.nav_stats  # Get computed navigation statistics
    
    # Use only a fraction of the dataset for testing
    if cfg.dataset_fraction < 1.0:
        subset_size = int(len(full_dataset) * cfg.dataset_fraction)
        indices = torch.randperm(len(full_dataset))[:subset_size]
        
        # Create new datasets with same normalization stats for consistency
        train_indices = indices[:int(0.8 * len(indices))]
        val_indices = indices[int(0.8 * len(indices)):]
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        print(f"Using {cfg.dataset_fraction*100:.0f}% of dataset for testing")
        print(f"Dataset split: {len(train_indices)} training, {len(val_indices)} validation samples")
    else:
        print(f"Using full dataset ({len(full_dataset)} samples)")
        # Split into train/validation (80-20 split of the full dataset)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, 
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, 
                            persistent_workers=True if cfg.num_workers > 0 else False,
                            prefetch_factor=cfg.prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False, 
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                            persistent_workers=True if cfg.num_workers > 0 else False,
                            prefetch_factor=cfg.prefetch_factor)
    
    # Store nav_stats for potential use in evaluation
    print_nav_stats_summary(nav_stats)
    
    # Save navigation statistics for later use
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nav_stats_path = os.path.join("logs", f"nav_stats_{timestamp}.json")
    save_nav_stats(nav_stats, nav_stats_path)
    
    # Create model with simple encoder for now (VGG has DataParallel issues)
    model = get_model(cfg, use_vgg=False).to(device)
    
    # Wrap model with DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = JointLoss(image_w=cfg.image_loss_weight, nav_w=cfg.nav_loss_weight, 
                        vq_w=cfg.vq_loss_weight, perceptual=False, device=device)
    
    # Add learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    
    # Setup CSV loggers
    train_csv_path, val_csv_path = setup_csv_loggers()
    print(f"Logging training to: {train_csv_path}")
    print(f"Logging validation to: {val_csv_path}")

    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(cfg.epochs):
        print(f"\n=== Epoch {epoch+1}/{cfg.epochs} ===")
        
        # Training phase
        model.train()
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        running_loss = 0.0
        epoch_train_loss = 0.0
        
        for iteration, batch in enumerate(pbar):
            imgs = batch['imgs'].to(device)        # (B,S,C,H,W)
            navs = batch['navs'].to(device)
            tgt_img = batch['tgt_img'].to(device)
            tgt_nav = batch['tgt_nav'].to(device)

            opt.zero_grad()
            recon, nav_pred, vq_loss = model(imgs, navs)
            loss, info = criterion(recon, tgt_img, nav_pred, tgt_nav, vq_loss)
            
            # Check for NaN or infinite loss
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss detected: {loss}")
                print(f"Image loss: {info['L_img']}, Nav loss: {info['L_nav']}, VQ loss: {info['vq']}")
                continue
            
            loss.backward()
            
            # Gradient clipping for stability
            if hasattr(cfg, 'max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            
            opt.step()

            # Handle multi-GPU case where loss might not be scalar
            loss_scalar = loss.mean().item() if loss.numel() > 1 else loss.item()
            running_loss += loss_scalar
            epoch_train_loss += loss_scalar
            
            # Get current learning rate
            current_lr = opt.param_groups[0]['lr']
            
            # Log training iteration to CSV
            log_training_iteration(train_csv_path, epoch, iteration, global_step, 
                                    loss_scalar, info, current_lr)
            
            # Update progress bar
            avg_loss = running_loss / (iteration + 1)
            pbar.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
            global_step += 1
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        print("Running validation...")
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics['total_loss']
        
        # Log validation to CSV
        log_validation_epoch(val_csv_path, epoch, val_metrics)
        
        # Print validation results
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"  - Image Loss: {val_metrics['img_loss']:.4f}")
        print(f"  - Nav Loss: {val_metrics['nav_loss']:.4f}")
        print(f"  - VQ Loss: {val_metrics['vq_loss']:.4f}")
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")
        
        # Handle DataParallel wrapper when saving
        model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        ckpt = {
            "model": model_state_dict,
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "cfg": vars(cfg)
        }
        
        # Save regular checkpoint
        save_checkpoint(ckpt, os.path.join(cfg.checkpoint_dir, f"ckpt_epoch{epoch}.pt"))
        
        # Save best model
        if is_best:
            save_checkpoint(ckpt, os.path.join(cfg.checkpoint_dir, "best_model.pt"))
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training log saved to: {train_csv_path}")
    print(f"Validation log saved to: {val_csv_path}")

if __name__ == "__main__":
    train()
