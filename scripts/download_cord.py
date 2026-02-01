#!/usr/bin/env python3
"""
Download and prepare CORD-v2 dataset for training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    from datasets import load_dataset
    
    print("Downloading CORD-v2 dataset from HuggingFace...")
    print("This may take a few minutes (~2GB download)")
    
    # Download dataset
    dataset = load_dataset("naver-clova-ix/cord-v2")
    
    # Print info
    print(f"\nDataset loaded successfully!")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")
    
    # Save locally
    output_dir = Path("data/cord")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset.save_to_disk(str(output_dir))
    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
