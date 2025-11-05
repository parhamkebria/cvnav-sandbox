#!/usr/bin/env python3
"""
Test script for coordinate conversion functions in utils.py
Demonstrates conversion between lat/lon/alt (degrees) and meters
"""

import numpy as np
from utils import deg_to_meters, meters_to_deg, haversine_distance, calculate_flight_metrics

def test_coordinate_conversion():
    """Test the coordinate conversion functions with sample data"""
    
    print("="*60)
    print("COORDINATE CONVERSION TESTING")
    print("="*60)
    
    # Sample GPS coordinates (similar to your flight data)
    sample_lats = [56.206139, 56.206400, 56.207135]
    sample_lons = [10.187765, 10.188591, 10.190989] 
    sample_alts = [15000, 20000, 25000]  # meters
    
    print("Original GPS Coordinates:")
    for i, (lat, lon, alt) in enumerate(zip(sample_lats, sample_lons, sample_alts)):
        print(f"  Point {i+1}: Lat={lat:.6f}°, Lon={lon:.6f}°, Alt={alt:.1f}m")
    
    print(f"\nReference Point (mean): Lat={np.mean(sample_lats):.6f}°, "
        f"Lon={np.mean(sample_lons):.6f}°, Alt={np.mean(sample_alts):.1f}m")
    
    # Convert to meters
    x_m, y_m, z_m = deg_to_meters(sample_lats, sample_lons, sample_alts)
    
    print(f"\nConverted to Local Meters (from reference point):")
    for i, (x, y, z) in enumerate(zip(x_m, y_m, z_m)):
        print(f"  Point {i+1}: X={x:.2f}m (East), Y={y:.2f}m (North), Z={z:.1f}m (Up)")
    
    # Test reverse conversion
    ref_lat, ref_lon, ref_alt = np.mean(sample_lats), np.mean(sample_lons), np.mean(sample_alts)
    converted_lats, converted_lons, converted_alts = meters_to_deg(
        x_m, y_m, z_m, ref_lat, ref_lon, ref_alt
    )
    
    print(f"\nReverse Conversion (should match original):")
    for i, (lat, lon, alt) in enumerate(zip(converted_lats, converted_lons, converted_alts)):
        print(f"  Point {i+1}: Lat={lat:.6f}°, Lon={lon:.6f}°, Alt={alt:.1f}m")
    
    # Test accuracy
    lat_error = np.max(np.abs(np.array(sample_lats) - converted_lats))
    lon_error = np.max(np.abs(np.array(sample_lons) - converted_lons))
    alt_error = np.max(np.abs(np.array(sample_alts) - converted_alts))
    
    print(f"\nConversion Accuracy:")
    print(f"  Max Latitude Error: {lat_error:.10f}° ({lat_error * 111320:.3f}m)")
    print(f"  Max Longitude Error: {lon_error:.10f}° ({lon_error * 85390:.3f}m)")  # Approx for lat ~56°
    print(f"  Max Altitude Error: {alt_error:.3f}m")
    
    # Test distance calculations
    print(f"\nDistance Calculations:")
    dist_01 = haversine_distance(sample_lats[0], sample_lons[0], 
                                sample_lats[1], sample_lons[1])
    dist_12 = haversine_distance(sample_lats[1], sample_lons[1],
                                sample_lats[2], sample_lons[2])
    print(f"  Distance Point 1->2: {dist_01:.2f}m")
    print(f"  Distance Point 2->3: {dist_12:.2f}m")
    
    # Test flight metrics
    metrics = calculate_flight_metrics(sample_lats, sample_lons, sample_alts)
    print(f"\nFlight Metrics:")
    print(f"  Total 2D Distance: {metrics['total_distance_2d']:.2f}m")
    print(f"  Total 3D Distance: {metrics['total_distance_3d']:.2f}m")
    print(f"  Max Distance from Start: {metrics['max_distance_from_start']:.2f}m")
    print(f"  Bounding Box: {metrics['bounding_box_meters'][0]:.2f}m × {metrics['bounding_box_meters'][1]:.2f}m")
    
    print("\n" + "="*60)
    print("✅ All coordinate conversion tests completed successfully!")
    print("="*60)


def test_with_flight_data():
    """Test with actual flight data if available"""
    try:
        # Try to import flight data reader
        import sys
        sys.path.append('.')
        from scripts.flight_path_visualizer import FlightDataReader
        
        print("\n" + "="*60)
        print("TESTING WITH REAL FLIGHT DATA")
        print("="*60)
        
        # Read a small sample of flight data
        reader = FlightDataReader()
        data_dir = "/home/parham/AuAir/data/annotations/annotation_files"
        
        # Read only first 100 points for quick test
        import os
        sample_files = [f for f in os.listdir(data_dir) if f.startswith("frame_") and f.endswith(".txt")][:100]
        
        sample_data = []
        for filename in sample_files:
            file_path = os.path.join(data_dir, filename)
            data = reader.parse_frame_file(file_path)
            if data and all(key in data for key in ['latitude', 'longitude', 'altitude']):
                sample_data.append(data)
            if len(sample_data) >= 10:  # Just test with 10 points
                break
        
        if sample_data:
            lats = [d['latitude'] for d in sample_data]
            lons = [d['longitude'] for d in sample_data] 
            alts = [d['altitude'] for d in sample_data]
            
            print(f"Testing with {len(sample_data)} real flight data points...")
            
            # Calculate metrics
            metrics = calculate_flight_metrics(lats, lons, alts)
            
            print(f"Real Flight Data Metrics:")
            print(f"  Coordinate Range: Lat {min(lats):.6f}° - {max(lats):.6f}°")
            print(f"                    Lon {min(lons):.6f}° - {max(lons):.6f}°")
            print(f"                    Alt {min(alts):.1f}m - {max(alts):.1f}m")
            print(f"  Total Distance (2D): {metrics['total_distance_2d']:.2f}m")
            print(f"  Total Distance (3D): {metrics['total_distance_3d']:.2f}m")
            print(f"  Bounding Box: {metrics['bounding_box_meters'][0]:.2f}m × {metrics['bounding_box_meters'][1]:.2f}m")
            
            print("✅ Real flight data test completed!")
        else:
            print("⚠️ No valid flight data found for testing")
            
    except Exception as e:
        print(f"⚠️ Could not test with real flight data: {e}")


if __name__ == "__main__":
    test_coordinate_conversion()
    test_with_flight_data()