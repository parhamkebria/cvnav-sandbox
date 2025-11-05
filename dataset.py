import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def parse_txt(txt_path):
    # Parse the structured annotation file format
    import re
    with open(txt_path,'r') as f:
        txt = f.read()
    
    # Extract specific values using regex patterns
    try:
        # Extract longitude
        lon_match = re.search(r'longtitude:\s*(-?\d+\.?\d*)', txt)
        lon = float(lon_match.group(1)) if lon_match else 0.0
        
        # Extract latitude  
        lat_match = re.search(r'latitude:\s*(-?\d+\.?\d*)', txt)
        lat = float(lat_match.group(1)) if lat_match else 0.0
        
        # Extract altitude
        alt_match = re.search(r'altitude:\s*(-?\d+\.?\d*)', txt)
        alt = float(alt_match.group(1)) if alt_match else 0.0
        
        # Extract angles (roll=phi, pitch=theta, yaw=psi)
        phi_match = re.search(r'angle_phi:\s*(-?\d+\.?\d*)', txt)
        roll = float(phi_match.group(1)) if phi_match else 0.0
        
        theta_match = re.search(r'angle_theta:\s*(-?\d+\.?\d*)', txt)
        pitch = float(theta_match.group(1)) if theta_match else 0.0
        
        psi_match = re.search(r'angle_psi:\s*(-?\d+\.?\d*)', txt)
        yaw = float(psi_match.group(1)) if psi_match else 0.0
        
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not parse {txt_path}: {e}")
        lat = lon = alt = roll = pitch = yaw = 0.0
    
    return np.array([lat, lon, alt, roll, pitch, yaw], dtype=np.float32)

class DroneSeqDataset(Dataset):
    def __init__(self, root_dir, seq_len=4, transform=None, annotation_dir=None, nav_stats=None):
        self.root = root_dir
        self.seq_len = seq_len
        # If annotation_dir is not provided, assume annotations are in the same directory as images
        self.annotation_dir = annotation_dir or root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir,"*.jpg")))
        
        # sequential naming time-ordered; build indices where we can get seq_len frames + next frame
        self.indices = []
        for i in range(len(self.files) - seq_len):
            self.indices.append(i)
        
        # Proper image transforms: ToTensor first, then normalize
        self.transform = transform or T.Compose([
            T.Resize((256, 448)),  # downsize to speed training
            T.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ])
        
        # Navigation normalization statistics
        # If not provided, compute from data or use reasonable defaults for Denmark drone flight
        if nav_stats is None:
            self.nav_stats = self._compute_nav_stats()
        else:
            self.nav_stats = nav_stats
            
        print(f"Navigation normalization stats:")
        for i, name in enumerate(['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw']):
            print(f"  {name}: mean={self.nav_stats['mean'][i]:.4f}, std={self.nav_stats['std'][i]:.4f}")
    
    def _compute_nav_stats(self):
        """Compute navigation statistics from a sample of the data"""
        print("Computing navigation statistics from data sample...")
        
        # Sample a subset of files to compute stats (for efficiency)
        sample_size = min(1000, len(self.files))
        sample_indices = np.linspace(0, len(self.files)-1, sample_size, dtype=int)
        
        nav_data = []
        for idx in sample_indices:
            try:
                base_filename = os.path.basename(os.path.splitext(self.files[idx])[0]) + ".txt"
                txt_path = os.path.join(self.annotation_dir, base_filename)
                nav = parse_txt(txt_path)
                nav_data.append(nav)
            except:
                continue
        
        if nav_data:
            nav_array = np.array(nav_data)
            mean = nav_array.mean(axis=0)
            std = nav_array.std(axis=0)
            
            # Ensure std is not zero (add small epsilon for numerical stability)
            std = np.where(std < 1e-6, 1.0, std)
        else:
            # Fallback to reasonable defaults for Denmark drone flight
            print("Warning: Could not compute stats from data, using defaults")
            mean = np.array([56.2, 10.2, 15000.0, 0.0, 0.0, 0.0])  # Denmark coordinates + reasonable altitude
            std = np.array([0.01, 0.01, 5000.0, 0.5, 0.5, 1.0])
        
        return {'mean': mean, 'std': std}
    
    def normalize_nav(self, nav):
        """Normalize navigation data"""
        return (nav - self.nav_stats['mean']) / self.nav_stats['std']
    
    def denormalize_nav(self, nav_normalized):
        """Denormalize navigation data"""
        return nav_normalized * self.nav_stats['std'] + self.nav_stats['mean']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        imgs = []
        navs = []
        for j in range(self.seq_len):
            img_path = self.files[i+j]
            # Get the base filename and look for corresponding txt file in annotation directory
            base_filename = os.path.basename(os.path.splitext(img_path)[0]) + ".txt"
            txt_path = os.path.join(self.annotation_dir, base_filename)
            
            # Load and transform image
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            
            # Load and normalize navigation data
            nav = parse_txt(txt_path)
            nav_normalized = self.normalize_nav(nav)
            
            imgs.append(img)
            navs.append(torch.from_numpy(nav_normalized.astype(np.float32)))
            
        # next frame (target)
        tgt_img_path = self.files[i + self.seq_len]
        # Get the base filename and look for corresponding txt file in annotation directory
        tgt_base_filename = os.path.basename(os.path.splitext(tgt_img_path)[0]) + ".txt"
        tgt_txt_path = os.path.join(self.annotation_dir, tgt_base_filename)
        
        # Load and transform target image
        tgt_img = Image.open(tgt_img_path).convert("RGB")
        tgt_img = self.transform(tgt_img)
        
        # Load and normalize target navigation data
        tgt_nav = parse_txt(tgt_txt_path)
        tgt_nav_normalized = self.normalize_nav(tgt_nav)

        imgs = torch.stack(imgs)           # (seq_len, C, H, W)
        navs = torch.stack(navs)           # (seq_len, 6) - normalized
        return {
            "imgs": imgs,         # input frames (normalized)
            "navs": navs,         # telemetry for input frames (normalized)
            "tgt_img": tgt_img,   # ground-truth next image (normalized)
            "tgt_nav": torch.from_numpy(tgt_nav_normalized.astype(np.float32)),   # ground-truth next nav (normalized)
        }

def make_dataloader(root, seq_len=4, batch_size=2, shuffle=True, num_workers=4, annotation_dir=None, sampler=None, nav_stats=None):
    ds = DroneSeqDataset(root, seq_len=seq_len, annotation_dir=annotation_dir, nav_stats=nav_stats)
    # If sampler is provided (e.g., DistributedSampler), don't shuffle
    if sampler is not None:
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                    pin_memory=True, sampler=sampler, persistent_workers=True if num_workers > 0 else False), ds.nav_stats
