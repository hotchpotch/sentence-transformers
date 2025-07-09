#!/usr/bin/env python
"""
Monitor small training completion and automatically start full training.
"""

import subprocess
import time
import os
import sys
from pathlib import Path

def is_process_running(process_name):
    """Check if a process is running by name."""
    try:
        result = subprocess.run(['pgrep', '-f', process_name], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def wait_for_small_completion():
    """Wait for small training to complete."""
    print("Waiting for small training to complete...")
    
    while True:
        if not is_process_running("train_pruning_only_small.py"):
            print("Small training completed!")
            break
        print("Small training still running... waiting 30 seconds")
        time.sleep(30)

def start_full_training():
    """Start full training."""
    print("Starting full training...")
    
    # Kill any existing full training process
    subprocess.run(['pkill', '-f', 'train_pruning_only_full.py'], 
                  capture_output=True)
    
    # Start new full training
    log_file = f"./log/train_pruning_only_full_background_{int(time.time())}.log"
    cmd = f"nohup uv run python scripts/train_pruning_only_full.py > {log_file} 2>&1 &"
    
    subprocess.run(cmd, shell=True)
    print(f"Full training started. Log: {log_file}")

def main():
    """Main function."""
    # Create log directory
    os.makedirs("./log", exist_ok=True)
    
    # Wait for small training to complete
    wait_for_small_completion()
    
    # Wait a bit for GPU memory to be freed
    print("Waiting 30 seconds for GPU memory to be freed...")
    time.sleep(30)
    
    # Start full training
    start_full_training()
    
    print("Full training monitor completed!")

if __name__ == "__main__":
    main()