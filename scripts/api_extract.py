#!/usr/bin/env python3
"""
API extraction script for web backend.
Uses GPU by default if available.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.pipeline import MenuPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-image", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    args = parser.parse_args()
    
    # Use GPU if available
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    
    # Create pipeline
    config = PipelineConfig(
        use_gpu=use_gpu,
        model_path=args.model
    )
    pipeline = MenuPipeline(config)
    
    # Process
    result = pipeline.process(args.image)
    
    # Save JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w') as f:
        f.write(result.to_json())
    
    # Save annotated image
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    pipeline.visualize(args.image, result, args.output_image)
    
    device = "GPU" if use_gpu else "CPU"
    print(f"[{device}] Extracted {len(result.ocr_results)} text elements in {result.processing_time_ms:.0f}ms")


if __name__ == "__main__":
    main()
