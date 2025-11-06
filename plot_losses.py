#!/usr/bin/env python3
"""
Plot training and validation losses from CSV log files.
Usage: python plot_losses.py [--session TIMESTAMP]
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
from datetime import datetime

def find_latest_session():
    """Find the most recent training session timestamp"""
    train_files = glob.glob("logs/training_loss_*.csv")
    if not train_files:
        return None
    
    # Extract timestamps and find the latest
    timestamps = []
    for f in train_files:
        # Extract timestamp from filename like "training_loss_20251106_124534.csv"
        basename = os.path.basename(f)
        timestamp = basename.replace("training_loss_", "").replace(".csv", "")
        timestamps.append(timestamp)
    
    return max(timestamps)

def load_csv_data(filepath):
    """Load CSV data into a dictionary of lists"""
    if not os.path.exists(filepath):
        return None
    
    data = {}
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            # Initialize data dict with empty lists
            for row in reader:
                if not data:  # First row, initialize columns
                    for key in row.keys():
                        data[key] = []
                
                # Add data from this row
                for key, value in row.items():
                    try:
                        # Try to convert to float, keep as string if it fails
                        data[key].append(float(value))
                    except ValueError:
                        data[key].append(value)
        
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_training_data(session_timestamp):
    """Load training and validation data for a given session"""
    train_file = f"logs/training_loss_{session_timestamp}.csv"
    val_file = f"logs/val_loss_{session_timestamp}.csv"
    
    data = {}
    
    # Load training data
    train_data = load_csv_data(train_file)
    if train_data:
        data['train'] = train_data
        print(f"Loaded training data: {len(train_data['total_loss'])} iterations")
    else:
        print(f"Training file not found or empty: {train_file}")
        data['train'] = None
    
    # Load validation data
    val_data = load_csv_data(val_file)
    if val_data:
        data['val'] = val_data
        print(f"Loaded validation data: {len(val_data['total_loss'])} epochs")
    else:
        print(f"Validation file not found or empty: {val_file}")
        data['val'] = None
    
    return data

def plot_training_losses(data, session_timestamp):
    """Plot training losses over iterations"""
    if data['train'] is None or len(data['train']['total_loss']) == 0:
        print("No training data to plot")
        return
    
    train_data = data['train']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Losses - Session {session_timestamp}', fontsize=16)
    
    # Plot 1: Total Loss
    axes[0, 0].plot(train_data['global_step'], train_data['total_loss'], 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Global Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add moving average for total loss
    if len(train_data['total_loss']) > 10:
        window = min(50, len(train_data['total_loss']) // 10)
        moving_avg = moving_average(train_data['total_loss'], window)
        if len(moving_avg) > 0:
            # Adjust x-axis for moving average
            x_ma = train_data['global_step'][window//2:window//2+len(moving_avg)]
            axes[0, 0].plot(x_ma, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            axes[0, 0].legend()
    
    # Plot 2: Component Losses
    axes[0, 1].plot(train_data['global_step'], train_data['img_loss'], 'g-', label='Image Loss', alpha=0.8)
    axes[0, 1].plot(train_data['global_step'], train_data['nav_loss'], 'orange', label='Nav Loss', alpha=0.8)
    axes[0, 1].plot(train_data['global_step'], train_data['vq_loss'], 'purple', label='VQ Loss', alpha=0.8)
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].set_xlabel('Global Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    if 'learning_rate' in train_data:
        axes[1, 0].plot(train_data['global_step'], train_data['learning_rate'], 'm-', linewidth=2)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Global Step')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Plot 4: Loss Distribution (Last 100 iterations)
    recent_losses = train_data['total_loss'][-min(100, len(train_data['total_loss'])):]
    axes[1, 1].hist(recent_losses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_title(f'Loss Distribution (Last {len(recent_losses)} iterations)')
    axes[1, 1].set_xlabel('Total Loss')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def moving_average(data, window):
    """Calculate moving average"""
    if len(data) < window:
        return []
    
    result = []
    for i in range(window, len(data) + 1):
        avg = sum(data[i-window:i]) / window
        result.append(avg)
    return result

def plot_validation_losses(data, session_timestamp):
    """Plot validation losses over epochs"""
    if data['val'] is None or len(data['val']['total_loss']) == 0:
        print("No validation data to plot")
        return None
    
    val_data = data['val']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Validation Losses - Session {session_timestamp}', fontsize=16)
    
    # Plot 1: Total Validation Loss
    axes[0].plot(val_data['epoch'], val_data['total_loss'], 'ro-', linewidth=2, markersize=6)
    axes[0].set_title('Validation Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Component Losses
    axes[1].plot(val_data['epoch'], val_data['img_loss'], 'go-', label='Image Loss', linewidth=2, markersize=4)
    axes[1].plot(val_data['epoch'], val_data['nav_loss'], 'o-', color='orange', label='Nav Loss', linewidth=2, markersize=4)
    axes[1].plot(val_data['epoch'], val_data['vq_loss'], 'mo-', label='VQ Loss', linewidth=2, markersize=4)
    axes[1].set_title('Validation Component Losses')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_combined_overview(data, session_timestamp):
    """Plot training and validation losses together"""
    if data['train'] is None:
        print("No training data for combined plot")
        return None
    
    train_data = data['train']
    val_data = data['val']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Training Overview - Session {session_timestamp}', fontsize=16)
    
    # Plot 1: Training loss with validation points
    axes[0].plot(train_data['global_step'], train_data['total_loss'], 'b-', alpha=0.6, linewidth=1, label='Training Loss')
    
    # Add moving average
    if len(train_data['total_loss']) > 10:
        window = min(50, len(train_data['total_loss']) // 10)
        moving_avg = moving_average(train_data['total_loss'], window)
        if len(moving_avg) > 0:
            x_ma = train_data['global_step'][window//2:window//2+len(moving_avg)]
            axes[0].plot(x_ma, moving_avg, 'b-', linewidth=2, label=f'Training MA ({window})')
    
    # Add validation points if available
    if val_data is not None and len(val_data['total_loss']) > 0:
        # Map validation epochs to approximate global steps
        max_step = max(train_data['global_step'])
        max_epoch = max(train_data['epoch']) if train_data['epoch'] else 0
        steps_per_epoch = max_step / (max_epoch + 1) if max_epoch > 0 else max_step
        val_steps = [(epoch + 1) * steps_per_epoch for epoch in val_data['epoch']]
        axes[0].plot(val_steps, val_data['total_loss'], 'ro', markersize=8, label='Validation Loss')
    
    axes[0].set_title('Total Loss (Training + Validation)')
    axes[0].set_xlabel('Global Step')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Component losses over time
    axes[1].plot(train_data['global_step'], train_data['img_loss'], 'g-', alpha=0.7, label='Image Loss')
    axes[1].plot(train_data['global_step'], train_data['nav_loss'], 'orange', alpha=0.7, label='Nav Loss')
    axes[1].plot(train_data['global_step'], train_data['vq_loss'], 'purple', alpha=0.7, label='VQ Loss')
    
    axes[1].set_title('Component Losses (Training)')
    axes[1].set_xlabel('Global Step')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_validation_losses(data, session_timestamp):
    """Plot validation losses over epochs"""
    if data['val'] is None or len(data['val']) == 0:
        print("No validation data to plot")
        return None
    
    val_df = data['val']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Validation Losses - Session {session_timestamp}', fontsize=16)
    
    # Plot 1: Total Validation Loss
    axes[0].plot(val_df['epoch'], val_df['total_loss'], 'ro-', linewidth=2, markersize=6)
    axes[0].set_title('Validation Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Component Losses
    axes[1].plot(val_df['epoch'], val_df['img_loss'], 'go-', label='Image Loss', linewidth=2, markersize=4)
    axes[1].plot(val_df['epoch'], val_df['nav_loss'], 'o-', color='orange', label='Nav Loss', linewidth=2, markersize=4)
    axes[1].plot(val_df['epoch'], val_df['vq_loss'], 'mo-', label='VQ Loss', linewidth=2, markersize=4)
    axes[1].set_title('Validation Component Losses')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_combined_overview(data, session_timestamp):
    """Plot training and validation losses together"""
    if data['train'] is None:
        print("No training data for combined plot")
        return None
    
    train_df = data['train']
    val_df = data['val']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Training Overview - Session {session_timestamp}', fontsize=16)
    
    # Plot 1: Training loss with validation points
    axes[0].plot(train_df['global_step'], train_df['total_loss'], 'b-', alpha=0.6, linewidth=1, label='Training Loss')
    
    # Add moving average
    if len(train_df) > 10:
        window = min(50, len(train_df) // 10)
        moving_avg = train_df['total_loss'].rolling(window=window, center=True).mean()
        axes[0].plot(train_df['global_step'], moving_avg, 'b-', linewidth=2, label=f'Training MA ({window})')
    
    # Add validation points if available
    if val_df is not None and len(val_df) > 0:
        # Map validation epochs to approximate global steps
        # Assume validation happens at end of each epoch
        steps_per_epoch = train_df['global_step'].max() / (train_df['epoch'].max() + 1) if len(train_df) > 0 else 1
        val_steps = (val_df['epoch'] + 1) * steps_per_epoch
        axes[0].plot(val_steps, val_df['total_loss'], 'ro', markersize=8, label='Validation Loss')
    
    axes[0].set_title('Total Loss (Training + Validation)')
    axes[0].set_xlabel('Global Step')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Component losses over time
    axes[1].plot(train_df['global_step'], train_df['img_loss'], 'g-', alpha=0.7, label='Image Loss')
    axes[1].plot(train_df['global_step'], train_df['nav_loss'], 'orange', alpha=0.7, label='Nav Loss')
    axes[1].plot(train_df['global_step'], train_df['vq_loss'], 'purple', alpha=0.7, label='VQ Loss')
    
    axes[1].set_title('Component Losses (Training)')
    axes[1].set_xlabel('Global Step')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_summary_stats(data, session_timestamp):
    """Print summary statistics about the training"""
    print(f"\n=== Training Summary - Session {session_timestamp} ===")
    
    if data['train'] is not None and len(data['train']['total_loss']) > 0:
        train_data = data['train']
        print(f"Training iterations: {len(train_data['total_loss'])}")
        print(f"Epochs completed: {max(train_data['epoch']) + 1}")
        print(f"Global steps: {max(train_data['global_step'])}")
        
        # Latest losses
        print(f"\nLatest Training Losses:")
        print(f"  Total: {train_data['total_loss'][-1]:.6f}")
        print(f"  Image: {train_data['img_loss'][-1]:.6f}")
        print(f"  Navigation: {train_data['nav_loss'][-1]:.6f}")
        print(f"  VQ: {train_data['vq_loss'][-1]:.6f}")
        if 'learning_rate' in train_data:
            print(f"  Learning Rate: {train_data['learning_rate'][-1]}")
        
        # Best (minimum) losses
        best_idx = train_data['total_loss'].index(min(train_data['total_loss']))
        print(f"\nBest Training Losses:")
        print(f"  Total: {train_data['total_loss'][best_idx]:.6f} (step {train_data['global_step'][best_idx]})")
        print(f"  Image: {train_data['img_loss'][best_idx]:.6f}")
        print(f"  Navigation: {train_data['nav_loss'][best_idx]:.6f}")
        print(f"  VQ: {train_data['vq_loss'][best_idx]:.6f}")
    
    if data['val'] is not None and len(data['val']['total_loss']) > 0:
        val_data = data['val']
        print(f"\nValidation epochs: {len(val_data['total_loss'])}")
        
        # Latest validation
        print(f"\nLatest Validation Losses:")
        print(f"  Total: {val_data['total_loss'][-1]:.6f}")
        print(f"  Image: {val_data['img_loss'][-1]:.6f}")
        print(f"  Navigation: {val_data['nav_loss'][-1]:.6f}")
        print(f"  VQ: {val_data['vq_loss'][-1]:.6f}")
        
        # Best validation
        best_val_idx = val_data['total_loss'].index(min(val_data['total_loss']))
        print(f"\nBest Validation Losses:")
        print(f"  Total: {val_data['total_loss'][best_val_idx]:.6f} (epoch {val_data['epoch'][best_val_idx]})")
        print(f"  Image: {val_data['img_loss'][best_val_idx]:.6f}")
        print(f"  Navigation: {val_data['nav_loss'][best_val_idx]:.6f}")
        print(f"  VQ: {val_data['vq_loss'][best_val_idx]:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Plot training and validation losses')
    parser.add_argument('--session', type=str, help='Session timestamp (e.g., 20251106_124534)')
    parser.add_argument('--save', action='store_true', help='Save plots as PNG files')
    parser.add_argument('--show', action='store_true', default=True, help='Show plots interactively')
    args = parser.parse_args()
    
    # Find session to plot
    if args.session:
        session_timestamp = args.session
    else:
        session_timestamp = find_latest_session()
        if session_timestamp is None:
            print("No training log files found in logs/ directory")
            return
        print(f"Using latest session: {session_timestamp}")
    
    # Load data
    data = load_training_data(session_timestamp)
    
    if data['train'] is None and data['val'] is None:
        print("No valid data found to plot")
        return
    
    # Print summary statistics
    print_summary_stats(data, session_timestamp)
    
    # Create plots
    plots = []
    
    # Training losses plot
    if data['train'] is not None:
        fig1 = plot_training_losses(data, session_timestamp)
        if fig1:
            plots.append(('training_losses', fig1))
    
    # Validation losses plot
    if data['val'] is not None:
        fig2 = plot_validation_losses(data, session_timestamp)
        if fig2:
            plots.append(('validation_losses', fig2))
    
    # Combined overview plot
    fig3 = plot_combined_overview(data, session_timestamp)
    if fig3:
        plots.append(('combined_overview', fig3))
    
    # Save plots if requested
    if args.save:
        for plot_name, fig in plots:
            filename = f"plots/{plot_name}_{session_timestamp}.png"
            os.makedirs("plots", exist_ok=True)
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
    
    # Show plots if requested
    if args.show and plots:
        plt.show()
    
    print(f"\nPlotting completed for session {session_timestamp}")

if __name__ == "__main__":
    main()