#!/usr/bin/env python3
"""
Quick test script to validate the entire framework on a small subset of data
"""

import os
import torch
from config import cfg
from dataset import DroneSeqDataset
from models import DronePredictor
from losses import JointLoss

def test_data_loading():
    """Test that data loading works correctly"""
    print("=== Testing Data Loading ===")
    
    try:
        dataset = DroneSeqDataset(cfg.data_root, seq_len=cfg.seq_len, annotation_dir=cfg.annotation_root)
        print(f"✅ Dataset loaded successfully: {len(dataset)} samples")
        
        # Test loading a single sample
        sample = dataset[0]
        print(f"✅ Sample shapes:")
        print(f"  - Input images: {sample['imgs'].shape}")
        print(f"  - Input navigation: {sample['navs'].shape}")
        print(f"  - Target image: {sample['tgt_img'].shape}")
        print(f"  - Target navigation: {sample['tgt_nav'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_creation():
    """Test that model can be created and performs forward pass"""
    print("\n=== Testing Model Creation ===")
    
    try:
        model = DronePredictor(cfg)
        print(f"✅ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None

def test_forward_pass(model):
    """Test forward pass with dummy data"""
    print("\n=== Testing Forward Pass ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy input
        batch_size = 1
        seq_len = cfg.seq_len
        C, H, W = 3, 256, 448  # From dataset resize
        
        imgs = torch.randn(batch_size, seq_len, C, H, W).to(device)
        navs = torch.randn(batch_size, seq_len, 6).to(device)
        
        print(f"✅ Input shapes: imgs={imgs.shape}, navs={navs.shape}")
        
        # Forward pass
        with torch.no_grad():
            recon, nav_pred, vq_loss = model(imgs, navs)
            
        print(f"✅ Forward pass successful:")
        print(f"  - Reconstructed image: {recon.shape}")
        print(f"  - Navigation prediction: {nav_pred.shape}")
        print(f"  - VQ loss: {vq_loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test loss computation"""
    print("\n=== Testing Loss Computation ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = JointLoss(image_w=cfg.image_loss_weight, nav_w=cfg.nav_loss_weight, 
                            perceptual=False, device=device)
        
        # Create dummy data
        batch_size = 1
        C, H, W = 3, 256, 448
        
        recon = torch.randn(batch_size, C, H, W).to(device)
        target = torch.randn(batch_size, C, H, W).to(device)
        nav_pred = torch.randn(batch_size, 6).to(device)
        nav_target = torch.randn(batch_size, 6).to(device)
        vq_loss = torch.tensor(0.5).to(device)
        
        loss, info = criterion(recon, target, nav_pred, nav_target, vq_loss)
        
        print(f"✅ Loss computation successful:")
        print(f"  - Total loss: {loss.item():.4f}")
        print(f"  - Image loss: {info['L_img']:.4f}")
        print(f"  - Navigation loss: {info['L_nav']:.4f}")
        print(f"  - VQ loss: {info['vq']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False

def test_training_step():
    """Test a single training step with real data"""
    print("\n=== Testing Training Step ===")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and data
        model = DronePredictor(cfg).to(device)
        criterion = JointLoss(image_w=cfg.image_loss_weight, nav_w=cfg.nav_loss_weight, 
                            perceptual=False, device=device)
        
        # Load real data
        dataset = DroneSeqDataset(cfg.data_root, seq_len=cfg.seq_len, annotation_dir=cfg.annotation_root)
        sample = dataset[0]
        
        # Prepare batch
        imgs = sample['imgs'].unsqueeze(0).to(device)
        navs = sample['navs'].unsqueeze(0).to(device)
        tgt_img = sample['tgt_img'].unsqueeze(0).to(device)
        tgt_nav = sample['tgt_nav'].unsqueeze(0).to(device)
        
        # Forward pass
        recon, nav_pred, vq_loss = model(imgs, navs)
        loss, info = criterion(recon, tgt_img, nav_pred, tgt_nav, vq_loss)
        
        # Backward pass
        loss.backward()
        
        print(f"✅ Training step successful:")
        print(f"  - Total loss: {loss.item():.4f}")
        print(f"  - Gradients computed successfully")
        
        return True
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Testing Drone Predictor Framework")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Dataset fraction: {cfg.dataset_fraction*100:.0f}%")
    print(f"Epochs configured: {cfg.epochs}")
    
    # Run all tests
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Loss Computation", test_loss_computation),
        ("Training Step", test_training_step),
    ]
    
    results = []
    model = None
    
    for test_name, test_func in tests:
        if test_name == "Model Creation":
            model = test_func()
            results.append((test_name, model is not None))
        elif test_name == "Forward Pass" and model is not None:
            result = test_func(model)
            results.append((test_name, result))
        else:
            result = test_func()
            results.append((test_name, result))
    
    # Summary
    print("\n" + "="*50)
    print("🎯 Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Framework is ready for training.")
        print("You can now run:")
        print("  python train.py  # For quick testing with 10% dataset")
        print("  # Change cfg.dataset_fraction = 1.0 for full training")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before training.")
    
    return all_passed

if __name__ == "__main__":
    main()