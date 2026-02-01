#!/usr/bin/env python3
"""
API extraction script for web backend.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MenuPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-image", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()
    
    # Create pipeline
    config = PipelineConfig(
        use_gpu=False,
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
    
    print(f"Extracted {len(result.ocr_results)} text elements")
    print(f"Processing time: {result.processing_time_ms:.1f}ms")


if __name__ == "__main__":
    main()
