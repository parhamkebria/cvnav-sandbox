import os
import torch
import math
import numpy as np
import json

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None):
    ck = torch.load(path, map_location='cpu')
    model.load_state_dict(ck['model'])
    if optimizer and 'optimizer' in ck:
        optimizer.load_state_dict(ck['optimizer'])
    return ck


# === GEOGRAPHIC COORDINATE CONVERSION FUNCTIONS ===

def deg_to_meters(lat_deg, lon_deg, alt_m, reference_lat=None, reference_lon=None, reference_alt=None):
    """
    Convert latitude/longitude/altitude from degrees to meters using a local reference point.
    
    Parameters:
    -----------
    lat_deg : float or array-like
        Latitude in degrees
    lon_deg : float or array-like  
        Longitude in degrees
    alt_m : float or array-like
        Altitude in meters (already in meters)
    reference_lat : float, optional
        Reference latitude in degrees. If None, uses the mean of input latitudes
    reference_lon : float, optional
        Reference longitude in degrees. If None, uses the mean of input longitudes
    reference_alt : float, optional
        Reference altitude in meters. If None, uses the mean of input altitudes
    
    Returns:
    --------
    tuple: (x_meters, y_meters, z_meters)
        x_meters: East-West distance in meters (positive = East)
        y_meters: North-South distance in meters (positive = North)  
        z_meters: Altitude difference in meters (positive = Up)
    """
    # Convert to numpy arrays for easier computation
    lat_deg = np.asarray(lat_deg)
    lon_deg = np.asarray(lon_deg)
    alt_m = np.asarray(alt_m)
    
    # Set reference point (use mean if not provided)
    if reference_lat is None:
        reference_lat = np.mean(lat_deg)
    if reference_lon is None:
        reference_lon = np.mean(lon_deg)
    if reference_alt is None:
        reference_alt = np.mean(alt_m)
    
    # Earth radius in meters
    R_EARTH = 6378137.0  # WGS84 equatorial radius
    
    # Convert degrees to radians
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    ref_lat_rad = np.radians(reference_lat)
    ref_lon_rad = np.radians(reference_lon)
    
    # Calculate differences in radians
    dlat_rad = lat_rad - ref_lat_rad
    dlon_rad = lon_rad - ref_lon_rad
    
    # Convert to meters using small angle approximation for local coordinates
    # This is accurate for small distances (< ~100km from reference point)
    x_meters = R_EARTH * np.cos(ref_lat_rad) * dlon_rad  # East-West
    y_meters = R_EARTH * dlat_rad                        # North-South
    z_meters = alt_m - reference_alt                     # Up-Down
    
    return x_meters, y_meters, z_meters


def meters_to_deg(x_meters, y_meters, z_meters, reference_lat, reference_lon, reference_alt):
    """
    Convert local meter coordinates back to latitude/longitude/altitude.
    
    Parameters:
    -----------
    x_meters : float or array-like
        East-West distance in meters from reference point
    y_meters : float or array-like
        North-South distance in meters from reference point
    z_meters : float or array-like
        Altitude difference in meters from reference altitude
    reference_lat : float
        Reference latitude in degrees
    reference_lon : float
        Reference longitude in degrees  
    reference_alt : float
        Reference altitude in meters
    
    Returns:
    --------
    tuple: (lat_deg, lon_deg, alt_m)
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees
        alt_m: Altitude in meters
    """
    # Convert to numpy arrays
    x_meters = np.asarray(x_meters)
    y_meters = np.asarray(y_meters)
    z_meters = np.asarray(z_meters)
    
    # Earth radius in meters
    R_EARTH = 6378137.0  # WGS84 equatorial radius
    
    # Convert reference to radians
    ref_lat_rad = np.radians(reference_lat)
    ref_lon_rad = np.radians(reference_lon)
    
    # Convert meters back to radians
    dlat_rad = y_meters / R_EARTH
    dlon_rad = x_meters / (R_EARTH * np.cos(ref_lat_rad))
    
    # Convert back to degrees
    lat_deg = np.degrees(ref_lat_rad + dlat_rad)
    lon_deg = np.degrees(ref_lon_rad + dlon_rad)
    alt_m = reference_alt + z_meters
    
    return lat_deg, lon_deg, alt_m


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth using Haversine formula.
    
    Parameters:
    -----------
    lat1, lon1 : float
        Latitude and longitude of first point in degrees
    lat2, lon2 : float
        Latitude and longitude of second point in degrees
    
    Returns:
    --------
    float: Distance in meters
    """
    # Earth radius in meters
    R_EARTH = 6378137.0
    
    # Convert to radians
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad, lon2_rad = np.radians([lat2, lon2])
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R_EARTH * c


def calculate_flight_metrics(latitudes, longitudes, altitudes):
    """
    Calculate various flight path metrics in meters.
    
    Parameters:
    -----------
    latitudes : array-like
        Latitude values in degrees
    longitudes : array-like
        Longitude values in degrees  
    altitudes : array-like
        Altitude values in meters
    
    Returns:
    --------
    dict: Dictionary containing flight metrics
        - total_distance_2d: Total 2D distance traveled in meters
        - total_distance_3d: Total 3D distance traveled in meters
        - max_distance_from_start: Maximum distance from starting point in meters
        - bounding_box_meters: Bounding box dimensions (width, height) in meters
        - reference_point: Reference lat/lon/alt used for calculations
    """
    # Convert to numpy arrays
    lats = np.array(latitudes)
    lons = np.array(longitudes)  
    alts = np.array(altitudes)
    
    # Convert to local meter coordinates
    x_m, y_m, z_m = deg_to_meters(lats, lons, alts)
    
    # Calculate total 2D and 3D distances
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    dz = np.diff(z_m)
    
    distances_2d = np.sqrt(dx**2 + dy**2)
    distances_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    
    total_distance_2d = np.sum(distances_2d)
    total_distance_3d = np.sum(distances_3d)
    
    # Maximum distance from starting point
    distances_from_start = np.sqrt(x_m**2 + y_m**2)
    max_distance_from_start = np.max(distances_from_start)
    
    # Bounding box in meters
    width_m = np.max(x_m) - np.min(x_m)
    height_m = np.max(y_m) - np.min(y_m)
    
    return {
        'total_distance_2d': total_distance_2d,
        'total_distance_3d': total_distance_3d, 
        'max_distance_from_start': max_distance_from_start,
        'bounding_box_meters': (width_m, height_m),
        'reference_point': (np.mean(lats), np.mean(lons), np.mean(alts)),
        'x_meters': x_m,
        'y_meters': y_m,
        'z_meters': z_m
    }


# === NAVIGATION DATA NORMALIZATION UTILITIES ===

def save_nav_stats(nav_stats, filepath):
    """Save navigation normalization statistics to file"""
    stats_dict = {
        'mean': nav_stats['mean'].tolist(),
        'std': nav_stats['std'].tolist(),
        'keys': ['latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw']
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"Navigation statistics saved to: {filepath}")

def load_nav_stats(filepath):
    """Load navigation normalization statistics from file"""
    with open(filepath, 'r') as f:
        stats_dict = json.load(f)
    
    nav_stats = {
        'mean': np.array(stats_dict['mean']),
        'std': np.array(stats_dict['std'])
    }
    
    print(f"Navigation statistics loaded from: {filepath}")
    return nav_stats

def denormalize_nav_batch(nav_normalized, nav_stats):
    """Denormalize a batch of navigation predictions"""
    # nav_normalized: (B, 6) tensor
    # nav_stats: dict with 'mean' and 'std' arrays
    
    if isinstance(nav_normalized, torch.Tensor):
        device = nav_normalized.device
        mean = torch.tensor(nav_stats['mean'], device=device, dtype=nav_normalized.dtype)
        std = torch.tensor(nav_stats['std'], device=device, dtype=nav_normalized.dtype)
        return nav_normalized * std + mean
    else:
        return nav_normalized * nav_stats['std'] + nav_stats['mean']

def normalize_nav_batch(nav, nav_stats):
    """Normalize a batch of navigation values"""
    # nav: (B, 6) tensor or array
    # nav_stats: dict with 'mean' and 'std' arrays
    
    if isinstance(nav, torch.Tensor):
        device = nav.device
        mean = torch.tensor(nav_stats['mean'], device=device, dtype=nav.dtype)
        std = torch.tensor(nav_stats['std'], device=device, dtype=nav.dtype)
        return (nav - mean) / std
    else:
        return (nav - nav_stats['mean']) / nav_stats['std']

def print_nav_stats_summary(nav_stats):
    """Print a summary of navigation statistics"""
    names = ['Latitude', 'Longitude', 'Altitude', 'Roll', 'Pitch', 'Yaw']
    units = ['°', '°', 'm', 'rad', 'rad', 'rad']
    
    print("\nNavigation Data Statistics:")
    print("=" * 60)
    for i, (name, unit) in enumerate(zip(names, units)):
        mean = nav_stats['mean'][i]
        std = nav_stats['std'][i]
        print(f"{name:10s}: mean={mean:10.4f} {unit}, std={std:8.4f} {unit}")
    print("=" * 60)
