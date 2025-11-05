#!/usr/bin/env python3
"""
Test script to validate data normalization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import DroneSeqDataset
from config import cfg
from utils import print_nav_stats_summary, denormalize_nav_batch

def test_normalization():
    """Test that normalization and denormalization work correctly"""
    print("=== Testing Data Normalization ===")
    
    # Create dataset
    dataset = DroneSeqDataset(cfg.data_root, seq_len=cfg.seq_len, annotation_dir=cfg.annotation_root)
    nav_stats = dataset.nav_stats
    
    print_nav_stats_summary(nav_stats)
    
    # Test a few samples
    print("\nTesting normalization/denormalization on samples:")
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        normalized_nav = sample['navs'][0]  # First frame navigation
        normalized_tgt = sample['tgt_nav']  # Target navigation
        
        # Denormalize
        denorm_nav = denormalize_nav_batch(normalized_nav.unsqueeze(0), nav_stats).squeeze(0)
        denorm_tgt = denormalize_nav_batch(normalized_tgt.unsqueeze(0), nav_stats).squeeze(0)
        
        print(f"\nSample {i}:")
        print(f"  Input nav (normalized):  {normalized_nav.numpy()}")
        print(f"  Input nav (denormalized): {denorm_nav.numpy()}")
        print(f"  Target nav (normalized): {normalized_tgt.numpy()}")
        print(f"  Target nav (denormalized): {denorm_tgt.numpy()}")
        
        # Check if normalized values have reasonable range (should be roughly [-3, 3] for most data)
        nav_range = normalized_nav.abs().max().item()
        tgt_range = normalized_tgt.abs().max().item()
        
        if nav_range > 10 or tgt_range > 10:
            print(f"  ⚠️  Warning: Large normalized values detected (max: {max(nav_range, tgt_range):.2f})")
        else:
            print(f"  ✅ Normalized values in reasonable range (max: {max(nav_range, tgt_range):.2f})")

def test_image_normalization():
    """Test image normalization"""
    print("\n=== Testing Image Normalization ===")
    
    dataset = DroneSeqDataset(cfg.data_root, seq_len=cfg.seq_len, annotation_dir=cfg.annotation_root)
    
    # Test a sample
    sample = dataset[0]
    imgs = sample['imgs']  # (seq_len, C, H, W)
    tgt_img = sample['tgt_img']  # (C, H, W)
    
    print(f"Image tensor shapes:")
    print(f"  Input images: {imgs.shape}")
    print(f"  Target image: {tgt_img.shape}")
    
    # Check value ranges (should be roughly [-2, 2] for ImageNet normalized images)
    img_min = imgs.min().item()
    img_max = imgs.max().item()
    tgt_min = tgt_img.min().item()
    tgt_max = tgt_img.max().item()
    
    print(f"Image value ranges:")
    print(f"  Input images: [{img_min:.3f}, {img_max:.3f}]")
    print(f"  Target image: [{tgt_min:.3f}, {tgt_max:.3f}]")
    
    if -3 <= img_min <= 3 and -3 <= img_max <= 3 and -3 <= tgt_min <= 3 and -3 <= tgt_max <= 3:
        print("  ✅ Image normalization looks correct (ImageNet normalized)")
    else:
        print("  ⚠️  Warning: Image values outside expected range for ImageNet normalization")

def analyze_dataset_statistics():
    """Analyze dataset to understand data distribution"""
    print("\n=== Dataset Statistics Analysis ===")
    
    dataset = DroneSeqDataset(cfg.data_root, seq_len=cfg.seq_len, annotation_dir=cfg.annotation_root)
    
    # Collect statistics from a sample of the data
    sample_size = min(100, len(dataset))
    nav_data = []
    
    print(f"Analyzing {sample_size} samples...")
    
    for i in range(sample_size):
        sample = dataset[i]
        # Get denormalized navigation data
        nav_normalized = sample['navs']  # (seq_len, 6)
        for j in range(nav_normalized.shape[0]):
            nav_denorm = denormalize_nav_batch(nav_normalized[j:j+1], dataset.nav_stats)
            nav_data.append(nav_denorm.numpy().flatten())
    
    nav_array = np.array(nav_data)  # (N, 6)
    
    names = ['Latitude', 'Longitude', 'Altitude', 'Roll', 'Pitch', 'Yaw']
    print(f"\nDataset statistics (from {len(nav_data)} samples):")
    print("-" * 60)
    
    for i, name in enumerate(names):
        values = nav_array[:, i]
        print(f"{name:10s}: min={values.min():8.4f}, max={values.max():8.4f}, "
              f"mean={values.mean():8.4f}, std={values.std():8.4f}")

def main():
    print("🔍 Data Normalization Test")
    print(f"Dataset fraction: {cfg.dataset_fraction}")
    
    try:
        test_image_normalization()
        test_normalization()
        analyze_dataset_statistics()
        
        print("\n✅ All normalization tests completed!")
        print("\nKey checks:")
        print("1. Images should be in range [-3, 3] (ImageNet normalized)")
        print("2. Navigation values should be in range [-3, 3] after normalization")
        print("3. Denormalization should recover original values")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()