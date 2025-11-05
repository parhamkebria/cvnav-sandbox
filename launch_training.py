#!/usr/bin/env python3
"""
Multi-GPU Training Launcher for Drone Predictor

This script automatically detects available GPUs and launches the appropriate training method:
- Single GPU: Regular training
- Multiple GPUs: DataParallel or DistributedDataParallel training
"""

import torch
import argparse
import os

def check_gpu_setup():
    """Check GPU availability and setup."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Training will use CPU.")
        return 0, "cpu"
    
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s):")
    
    for i in range(gpu_count):
        gpu_props = torch.cuda.get_device_properties(i)
        memory_gb = gpu_props.total_memory / (1024**3)
        print(f"  GPU {i}: {gpu_props.name} ({memory_gb:.1f} GB)")
    
    return gpu_count, "cuda"

def main():
    parser = argparse.ArgumentParser(description='Launch multi-GPU training')
    parser.add_argument('--method', choices=['auto', 'dataparallel', 'distributed'], 
                        default='auto', help='Multi-GPU training method')
    parser.add_argument('--gpus', type=str, default=None, 
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    args = parser.parse_args()
    
    # Set visible GPUs if specified
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f"Using GPUs: {args.gpus}")
    
    gpu_count, device_type = check_gpu_setup()
    
    if gpu_count == 0:
        print("Training on CPU...")
        from train import train
        train()
    elif gpu_count == 1:
        print("Training on single GPU...")
        from train import train
        train()
    else:
        # Multiple GPUs available
        if args.method == 'auto':
            # Choose best method automatically
            if gpu_count >= 2:
                method = 'distributed'  # DDP is generally better for 2+ GPUs
            else:
                method = 'dataparallel'
        else:
            method = args.method
        
        print(f"Training on {gpu_count} GPUs using {method} method...")
        
        if method == 'distributed':
            from train_distributed import train_distributed
            success = train_distributed()
            if not success:
                print("Falling back to DataParallel...")
                from train import train
                train()
        else:  # dataparallel
            from train import train
            train()

if __name__ == "__main__":
    main()