#!/usr/bin/env python3
"""
Menu OCR - Extract structured data from restaurant menu images.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import MenuPipeline, PipelineConfig, extract_menu


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured data from menu images"
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to menu image"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSON file path"
    )
    parser.add_argument(
        "-v", "--visualize",
        type=Path,
        help="Save annotated image to this path"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for processing"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)"
    )
    
    args = parser.parse_args()
    
    if not args.image.exists():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    
    # Extract menu
    result = extract_menu(
        args.image,
        output_json=args.output,
        output_image=args.visualize,
        use_gpu=args.gpu
    )
    
    # Print result
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))
    
    if args.output:
        print(f"\nSaved to: {args.output}", file=sys.stderr)
    if args.visualize:
        print(f"Annotated image: {args.visualize}", file=sys.stderr)


if __name__ == "__main__":
    main()
