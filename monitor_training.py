#!/usr/bin/env python3
"""
Real-time loss monitoring script. Watches log files and updates plots automatically.
Usage: python monitor_training.py [--session TIMESTAMP] [--refresh SECONDS]
"""

import time
import os
import argparse
from plot_losses import load_training_data, find_latest_session, plot_combined_overview
import matplotlib.pyplot as plt

def monitor_training(session_timestamp=None, refresh_interval=30):
    """Monitor training in real-time"""
    if session_timestamp is None:
        session_timestamp = find_latest_session()
        if session_timestamp is None:
            print("No training session found")
            return
    
    print(f"Monitoring session: {session_timestamp}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    
    plt.ion()  # Turn on interactive mode
    fig = None
    
    try:
        while True:
            # Load latest data
            data = load_training_data(session_timestamp)
            
            if data['train'] is not None and len(data['train']['total_loss']) > 0:
                # Clear previous plot
                plt.clf()
                
                # Create new plot
                fig = plot_combined_overview(data, session_timestamp)
                
                if fig:
                    # Add timestamp to title
                    current_time = time.strftime("%H:%M:%S")
                    fig.suptitle(f'Training Monitor - Session {session_timestamp} (Updated: {current_time})', fontsize=16)
                    
                    # Show latest stats in title
                    train_data = data['train']
                    latest_loss = train_data['total_loss'][-1]
                    iterations = len(train_data['total_loss'])
                    
                    print(f"\r[{current_time}] Iterations: {iterations}, Latest Loss: {latest_loss:.6f}", end="", flush=True)
                    
                    plt.draw()
                    plt.pause(0.1)
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] Waiting for training data...", end="", flush=True)
            
            # Wait before next update
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
        plt.ioff()  # Turn off interactive mode
        if fig:
            plt.show()  # Keep final plot open

def main():
    parser = argparse.ArgumentParser(description='Monitor training losses in real-time')
    parser.add_argument('--session', type=str, help='Session timestamp to monitor')
    parser.add_argument('--refresh', type=int, default=30, help='Refresh interval in seconds (default: 30)')
    args = parser.parse_args()
    
    monitor_training(args.session, args.refresh)

if __name__ == "__main__":
    main()