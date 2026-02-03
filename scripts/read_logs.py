import os
import sys
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def main(log_path):
    print(f"Reading logs from: {log_path}")
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    target_tags = [
        'train/loss', 
        'train/value_loss', 
        'train/policy_gradient_loss', 
        'train/entropy_loss',
        'train/std',
        'train/explained_variance',
        'time/fps'
    ]
    
    for tag in target_tags:
        if tag in tags:
            events = ea.Scalars(tag)
            print(f"\n--- {tag} ---")
            # 最初、真ん中、最後を表示
            indices = [0, len(events)//2, -1]
            for i in sorted(list(set(indices))):
                if i < len(events):
                    e = events[i]
                    print(f"Index {i}, Step: {e.step}, Value: {e.value:.6f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 read_logs.py <path_to_tfevents>")
    else:
        main(sys.argv[1])
