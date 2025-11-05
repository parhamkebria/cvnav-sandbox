#!/usr/bin/env python3

import os
import re
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime, timedelta
import argparse

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import coordinate conversion functions
from utils import deg_to_meters, calculate_flight_metrics

# Try to import Plotly for interactive 3D plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Plotly not installed. Install with: pip install plotly")
    print("Interactive 3D plot will not be created.")
    PLOTLY_AVAILABLE = False


class FlightDataReader:
    def __init__(self):
        self.flight_data = []
    
    def parse_frame_file(self, file_path):
        """Parse a single frame annotation file and extract GPS data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Extract data using regex patterns
            data = {}
            
            # Extract image name
            image_name_match = re.search(r'image_name:\s*(.+)', content)
            if image_name_match:
                data['image_name'] = image_name_match.group(1).strip()
            
            # Extract platform
            platform_match = re.search(r'platform:\s*(.+)', content)
            if platform_match:
                data['platform'] = platform_match.group(1).strip()
            
            # Extract time components
            year_match = re.search(r'year:\s*(\d+)', content)
            month_match = re.search(r'month:\s*(\d+)', content)
            day_match = re.search(r'day:\s*(\d+)', content)
            hour_match = re.search(r'hour:\s*(\d+)', content)
            min_match = re.search(r'min:\s*(\d+)', content)
            sec_match = re.search(r'sec:\s*(\d+)', content)
            ms_match = re.search(r'ms:\s*([\d.]+)', content)
            
            if all([year_match, month_match, day_match, hour_match, min_match, sec_match, ms_match]):
                year = int(year_match.group(1))
                month = int(month_match.group(1))
                day = int(day_match.group(1))
                hour = int(hour_match.group(1))
                minute = int(min_match.group(1))
                second = int(sec_match.group(1))
                millisecond = float(ms_match.group(1))
                
                # Handle millisecond conversion properly
                # Convert to microseconds and ensure it's within valid range
                if millisecond >= 1000:
                    # If milliseconds is > 1000, it might be in a different unit
                    # Convert to seconds and add to the second field
                    additional_seconds = int(millisecond // 1000)
                    remaining_ms = millisecond % 1000
                    microseconds = int(remaining_ms * 1000)
                    
                    # Add additional seconds to the timestamp
                    base_datetime = datetime(year, month, day, hour, minute, second)
                    data['timestamp'] = base_datetime + timedelta(seconds=additional_seconds, 
                                                                microseconds=microseconds)
                else:
                    # Normal case where ms is < 1000
                    microseconds = int(millisecond * 1000)
                    data['timestamp'] = datetime(year, month, day, hour, minute, second, microseconds)
            
            # Extract GPS coordinates
            longitude_match = re.search(r'longtitude:\s*([-\d.]+)', content)
            latitude_match = re.search(r'latitude:\s*([-\d.]+)', content)
            altitude_match = re.search(r'altitude:\s*([-\d.]+)', content)
            
            if longitude_match:
                data['longitude'] = float(longitude_match.group(1))
            if latitude_match:
                data['latitude'] = float(latitude_match.group(1))
            if altitude_match:
                data['altitude'] = float(altitude_match.group(1))
            
            # Extract linear velocities
            linear_x_match = re.search(r'linear_x:\s*([-\d.]+)', content)
            linear_y_match = re.search(r'linear_y:\s*([-\d.]+)', content)
            linear_z_match = re.search(r'linear_z:\s*([-\d.]+)', content)
            
            if linear_x_match:
                data['linear_x'] = float(linear_x_match.group(1))
            if linear_y_match:
                data['linear_y'] = float(linear_y_match.group(1))
            if linear_z_match:
                data['linear_z'] = float(linear_z_match.group(1))
            
            # Extract angles
            angle_phi_match = re.search(r'angle_phi:\s*([-\d.]+)', content)
            angle_theta_match = re.search(r'angle_theta:\s*([-\d.]+)', content)
            angle_psi_match = re.search(r'angle_psi:\s*([-\d.]+)', content)
            
            if angle_phi_match:
                data['angle_phi'] = float(angle_phi_match.group(1))
            if angle_theta_match:
                data['angle_theta'] = float(angle_theta_match.group(1))
            if angle_psi_match:
                data['angle_psi'] = float(angle_psi_match.group(1))
            
            return data
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return None
    
    def read_directory(self, directory_path):
        """Read all frame files from the specified directory"""
        print(f"Attempting to read directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        print(f"Directory exists. Listing contents...")
        all_files = os.listdir(directory_path)
        print(f"Total files in directory: {len(all_files)}")
        
        # Find all frame files matching the pattern
        frame_files = []
        for filename in all_files:
            if filename.startswith("frame_") and filename.endswith(".txt"):
                frame_files.append(filename)
        
        # Sort files to ensure chronological order
        frame_files.sort()
        
        # Parse each file
        for filename in frame_files:
            file_path = os.path.join(directory_path, filename)
            data = self.parse_frame_file(file_path)
            
            if data and all(key in data for key in ['latitude', 'longitude', 'altitude']):
                self.flight_data.append(data)
            else:
                print(f"Warning: Incomplete data in {filename}")
        
        print(f"Successfully parsed {len(self.flight_data)} frames with GPS data")
        
        # Sort by timestamp if available
        if self.flight_data and 'timestamp' in self.flight_data[0]:
            self.flight_data.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        return self.flight_data


class FlightPathVisualizer:
    def __init__(self, flight_data):
        self.flight_data = flight_data
        
        # Convert coordinates to meters for all plots
        if self.flight_data:
            self._convert_to_meters()
    
    def _convert_to_meters(self):
        """Convert lat/lon coordinates to meters using local coordinate system"""
        # Extract original coordinates
        latitudes = [data['latitude'] for data in self.flight_data]
        longitudes = [data['longitude'] for data in self.flight_data]
        altitudes = [data['altitude'] for data in self.flight_data]
        
        # Convert to meters using utils function
        x_meters, y_meters, z_meters = deg_to_meters(latitudes, longitudes, altitudes)
        
        # Store the reference point for display
        self.reference_lat = np.mean(latitudes)
        self.reference_lon = np.mean(longitudes)
        self.reference_alt = np.mean(altitudes)
        
        # Add meter coordinates to flight data
        for i, data in enumerate(self.flight_data):
            data['x_meters'] = x_meters[i]
            data['y_meters'] = y_meters[i] 
            data['z_meters'] = z_meters[i]
        
        print(f"Coordinate conversion completed:")
        print(f"Reference point: Lat={self.reference_lat:.6f}°, Lon={self.reference_lon:.6f}°, Alt={self.reference_alt:.1f}m")
        print(f"Meter coordinate ranges:")
        print(f"  X (East-West): {min(x_meters):.2f}m to {max(x_meters):.2f}m")
        print(f"  Y (North-South): {min(y_meters):.2f}m to {max(y_meters):.2f}m") 
        print(f"  Z (Altitude): {min(z_meters):.1f}m to {max(z_meters):.1f}m")
    
    def create_3d_plot(self, save_plot=False, output_filename="flight_path_3d.png"):
        """Create a 3D plot of the flight path in meters"""
        if not self.flight_data:
            print("No flight data available for plotting")
            return
        
        # Extract meter coordinates
        x_meters = [data['x_meters'] for data in self.flight_data]
        y_meters = [data['y_meters'] for data in self.flight_data]
        z_meters = [data['z_meters'] for data in self.flight_data]
        altitudes = [data['altitude'] for data in self.flight_data]  # For color coding
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create the flight path as a line
        ax.plot(x_meters, y_meters, z_meters, 'b-', linewidth=2, alpha=0.7, label='Flight Path')
        
        # Mark start and end points
        ax.scatter(x_meters[0], y_meters[0], z_meters[0], 
                    c='green', s=100, marker='o', label='Start')
        ax.scatter(x_meters[-1], y_meters[-1], z_meters[-1], 
                    c='red', s=100, marker='s', label='End')
        
        # Color code points by altitude for better visualization
        scatter = ax.scatter(x_meters, y_meters, z_meters, 
                            c=altitudes, cmap='viridis', s=20, alpha=0.6)
        
        # Add colorbar for altitude
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Altitude (m)', rotation=270, labelpad=20)
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Flight Path Visualization (Meter Coordinates)')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Print statistics
        print(f"\nFlight Statistics (Meter Coordinates):")
        print(f"Total points: {len(self.flight_data)}")
        print(f"X range: {min(x_meters):.1f}m to {max(x_meters):.1f}m")
        print(f"Y range: {min(y_meters):.1f}m to {max(y_meters):.1f}m")
        print(f"Z range: {min(z_meters):.1f}m to {max(z_meters):.1f}m")
        print(f"Altitude range: {min(altitudes):.1f}m to {max(altitudes):.1f}m")
        print(f"Altitude variation: {max(altitudes) - min(altitudes):.1f}m")
        
        if save_plot:
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"3D plot saved as {output_filename}")
        
        plt.close()
    
    def create_2d_plots(self, save_plot=False):
        """Create additional 2D plots for better analysis in meter coordinates"""
        if not self.flight_data:
            print("No flight data available for plotting")
            return
        
        # Extract meter coordinates
        x_meters = [data['x_meters'] for data in self.flight_data]
        y_meters = [data['y_meters'] for data in self.flight_data]
        z_meters = [data['z_meters'] for data in self.flight_data]
        altitudes = [data['altitude'] for data in self.flight_data]
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Top-down view (X vs Y in meters)
        ax1.plot(x_meters, y_meters, 'b-', linewidth=1, alpha=0.7)
        scatter1 = ax1.scatter(x_meters, y_meters, c=altitudes, cmap='viridis', s=10)
        ax1.scatter(x_meters[0], y_meters[0], c='green', s=50, marker='o', label='Start')
        ax1.scatter(x_meters[-1], y_meters[-1], c='red', s=50, marker='s', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Top-down View (colored by altitude)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1)
        
        # 2. Z-coordinate (altitude) profile over time
        frame_numbers = list(range(len(z_meters)))
        ax2.plot(frame_numbers, z_meters, 'g-', linewidth=2)
        ax2.fill_between(frame_numbers, z_meters, alpha=0.3, color='green')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Z-coordinate Profile')
        ax2.grid(True, alpha=0.3)
        
        # 3. Y vs Z coordinate profile
        ax3.plot(y_meters, z_meters, 'r-', linewidth=1, alpha=0.7)
        ax3.scatter(y_meters, z_meters, c=frame_numbers, cmap='plasma', s=10)
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Y vs Z Profile')
        ax3.grid(True, alpha=0.3)
        
        # 4. X vs Z coordinate profile
        ax4.plot(x_meters, z_meters, 'm-', linewidth=1, alpha=0.7)
        ax4.scatter(x_meters, z_meters, c=frame_numbers, cmap='plasma', s=10)
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Z (m)')
        ax4.set_title('X vs Z Profile')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('flight_analysis_2d.png', dpi=300, bbox_inches='tight')
            print("2D analysis plots saved as flight_analysis_2d.png")
        
        plt.close()
    
    def create_interactive_3d_plot(self, output_filename="flight_path_interactive_3d.html"):
        """Create an interactive 3D plot using Plotly that can be rotated with mouse (meter coordinates)"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Cannot create interactive 3D plot.")
            return
            
        if not self.flight_data:
            print("No flight data available for plotting")
            return
        
        # Extract meter coordinates
        x_meters = [data['x_meters'] for data in self.flight_data]
        y_meters = [data['y_meters'] for data in self.flight_data]
        z_meters = [data['z_meters'] for data in self.flight_data]
        altitudes = [data['altitude'] for data in self.flight_data]
        
        # Create frame numbers for time sequence
        frame_numbers = list(range(len(self.flight_data)))
        
        # Create the interactive 3D plot
        fig = go.Figure()
        
        # Add flight path as a 3D line
        fig.add_trace(go.Scatter3d(
            x=x_meters,
            y=y_meters, 
            z=z_meters,
            mode='lines+markers',
            line=dict(
                color=altitudes,
                colorscale='Viridis',
                width=4,
                colorbar=dict(
                    title="Altitude (m)",
                    x=0.9,  # Position colorbar on the right side
                    xanchor="left",
                    thickness=15,
                    len=0.8
                )
            ),
            marker=dict(
                size=2,
                color=altitudes,
                colorscale='Viridis',
                opacity=0.8
            ),
            name='Flight Path',
            text=[f'Frame: {i}<br>X: {x:.1f}m<br>Y: {y:.1f}m<br>Z: {z:.1f}m<br>Alt: {alt:.1f}m' 
                    for i, x, y, z, alt in zip(frame_numbers, x_meters, y_meters, z_meters, altitudes)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add start point
        fig.add_trace(go.Scatter3d(
            x=[x_meters[0]],
            y=[y_meters[0]],
            z=[z_meters[0]],
            mode='markers',
            marker=dict(size=8, color='green'),
            name='Start Point',
            text=[f'START<br>X: {x_meters[0]:.1f}m<br>Y: {y_meters[0]:.1f}m<br>Z: {z_meters[0]:.1f}m<br>Alt: {altitudes[0]:.1f}m'],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add end point
        fig.add_trace(go.Scatter3d(
            x=[x_meters[-1]],
            y=[y_meters[-1]],
            z=[z_meters[-1]],
            mode='markers',
            marker=dict(size=8, color='red'),
            name='End Point',
            text=[f'END<br>X: {x_meters[-1]:.1f}m<br>Y: {y_meters[-1]:.1f}m<br>Z: {z_meters[-1]:.1f}m<br>Alt: {altitudes[-1]:.1f}m'],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': '3D Interactive Flight Path Visualization (Meter Coordinates)',
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                domain=dict(x=[0.15, 0.85], y=[0, 1])  # Make room for legend on left and colorbar on right
            ),
            legend=dict(
                x=0.01,  # Position legend on the far left
                y=0.99,  # Position at top
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent white background
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=1
            ),
            width=1200,  # Increased width to accommodate legend and colorbar
            height=800,
            margin=dict(l=150, r=100, t=50, b=50)  # Increased left margin for legend, right for colorbar
        )
        
        # Save the interactive plot
        fig.write_html(output_filename)
        print(f"Interactive 3D plot saved as {output_filename}")
        print("Open the HTML file in your web browser to interact with the 3D plot!")
    
    def create_time_series_plots(self, save_plot=False, output_filename="flight_time_series.png"):
        """Create separate time series plots for X, Y, Z coordinates vs time"""
        if not self.flight_data:
            print("No flight data available for plotting")
            return
        
        # Extract meter coordinates
        x_meters = [data['x_meters'] for data in self.flight_data]
        y_meters = [data['y_meters'] for data in self.flight_data]
        z_meters = [data['z_meters'] for data in self.flight_data]
        altitudes = [data['altitude'] for data in self.flight_data]
        
        # Extract timestamps if available, otherwise use frame numbers
        if 'timestamp' in self.flight_data[0]:
            timestamps = [data['timestamp'] for data in self.flight_data]
            x_label = 'Time'
            x_data = timestamps
        else:
            frame_numbers = list(range(len(self.flight_data)))
            x_label = 'Frame Number'
            x_data = frame_numbers
        
        # Create figure with 3 subplots (3 rows, 1 column)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Flight Data Time Series (Meter Coordinates)', fontsize=16, fontweight='bold')
        
        # Plot 1: Z-coordinate (altitude) vs Time
        ax1.plot(x_data, z_meters, 'b-', linewidth=2, alpha=0.8)
        ax1.fill_between(x_data, z_meters, alpha=0.3, color='blue')
        ax1.set_ylabel('Z (m)', fontsize=12)
        ax1.set_title('Z-coordinate over Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45 if 'timestamp' in self.flight_data[0] else 0)
        
        # Add Z-coordinate statistics
        z_min, z_max = min(z_meters), max(z_meters)
        z_mean = sum(z_meters) / len(z_meters)
        ax1.axhline(y=z_mean, color='red', linestyle='--', alpha=0.7, 
                    label=f'Mean: {z_mean:.1f}m')
        ax1.legend()
        
        # Plot 2: Y-coordinate vs Time
        ax2.plot(x_data, y_meters, 'g-', linewidth=2, alpha=0.8)
        ax2.fill_between(x_data, y_meters, alpha=0.3, color='green')
        ax2.set_ylabel('Y (m)', fontsize=12)
        ax2.set_title('Y-coordinate over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45 if 'timestamp' in self.flight_data[0] else 0)
        
        # Add Y-coordinate statistics
        y_min, y_max = min(y_meters), max(y_meters)
        y_mean = sum(y_meters) / len(y_meters)
        ax2.axhline(y=y_mean, color='red', linestyle='--', alpha=0.7,
                    label=f'Mean: {y_mean:.1f}m')
        ax2.legend()
        
        # Plot 3: X-coordinate vs Time
        ax3.plot(x_data, x_meters, 'orange', linewidth=2, alpha=0.8)
        ax3.fill_between(x_data, x_meters, alpha=0.3, color='orange')
        ax3.set_xlabel(x_label, fontsize=12)
        ax3.set_ylabel('X (m)', fontsize=12)
        ax3.set_title('X-coordinate over Time', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45 if 'timestamp' in self.flight_data[0] else 0)
        
        # Add X-coordinate statistics
        x_min, x_max = min(x_meters), max(x_meters)
        x_mean = sum(x_meters) / len(x_meters)
        ax3.axhline(y=x_mean, color='red', linestyle='--', alpha=0.7,
                    label=f'Mean: {x_mean:.1f}m')
        ax3.legend()
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        # Print time series statistics
        print(f"\nTime Series Statistics (Meter Coordinates):")
        print(f"Total data points: {len(self.flight_data)}")
        if 'timestamp' in self.flight_data[0]:
            duration = timestamps[-1] - timestamps[0]
            print(f"Flight duration: {duration}")
            print(f"Start time: {timestamps[0]}")
            print(f"End time: {timestamps[-1]}")
        
        print(f"X-coordinate: {x_min:.1f}m - {x_max:.1f}m (mean: {x_mean:.1f}m)")
        print(f"Y-coordinate: {y_min:.1f}m - {y_max:.1f}m (mean: {y_mean:.1f}m)")
        print(f"Z-coordinate: {z_min:.1f}m - {z_max:.1f}m (mean: {z_mean:.1f}m)")
        alt_min, alt_max = min(altitudes), max(altitudes)
        alt_mean = sum(altitudes) / len(altitudes)
        print(f"Altitude: {alt_min:.1f}m - {alt_max:.1f}m (mean: {alt_mean:.1f}m)")
        
        if save_plot:
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"Time series plots saved as {output_filename}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize flight path from frame annotation files')
    parser.add_argument('directory', help='Directory containing frame_*.txt files')
    parser.add_argument('--save', action='store_true', help='Save plots to files')
    parser.add_argument('--2d', action='store_true', help='Also create 2D analysis plots')
    parser.add_argument('--time-series', action='store_true', help='Also create time series plots')
    
    args = parser.parse_args()
    
    try:
        # Read flight data
        reader = FlightDataReader()
        flight_data = reader.read_directory(args.directory)
        
        if not flight_data:
            print("No valid flight data found!")
            return
        
        # Create visualizations
        visualizer = FlightPathVisualizer(flight_data)
        
        # Create 3D plot (always save)
        visualizer.create_3d_plot(save_plot=True)
        
        # Create interactive 3D plot (always save)
        visualizer.create_interactive_3d_plot()
        
        # Create time series plots if requested (always save)
        if getattr(args, 'time_series', False):
            visualizer.create_time_series_plots(save_plot=True)
        
        # Create 2D plots if requested (always save)
        if getattr(args, '2d', False):
            visualizer.create_2d_plots(save_plot=True)
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # If run without arguments, use default directory
    if len(os.sys.argv) == 1:
        default_dir = "/home/parham/AuAir/data/annotations/annotation_files"
        print(f"No directory specified, using default: {default_dir}")
        
        reader = FlightDataReader()
        flight_data = reader.read_directory(default_dir)
        
        if flight_data:
            visualizer = FlightPathVisualizer(flight_data)
            # Always save 3D plot
            visualizer.create_3d_plot(save_plot=True)
            
            # Create interactive 3D plot
            print("Creating interactive 3D plot...")
            visualizer.create_interactive_3d_plot()
            
            # Create time series plots
            print("Creating time series plots...")
            visualizer.create_time_series_plots(save_plot=True)
            
            # Also create and save 2D plots
            print("Creating 2D analysis plots...")
            visualizer.create_2d_plots(save_plot=True)
        else:
            print("No valid flight data found!")
    else:
        main()