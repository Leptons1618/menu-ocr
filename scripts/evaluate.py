#!/usr/bin/env python3
"""
Evaluate menu extraction on sample data.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MenuPipeline, PipelineConfig


def load_ground_truth(json_path: Path) -> dict:
    """Load ground truth JSON."""
    with open(json_path) as f:
        return json.load(f)


def extract_items(menu_json: dict) -> list[dict]:
    """Extract flat list of items from menu structure."""
    items = []
    for section in menu_json.get("menu", []):
        for group in section.get("groups", []):
            for item in group.get("items", []):
                items.append({
                    "name": item.get("name", "").lower().strip(),
                    "price": item.get("price"),
                })
    return items


def fuzzy_match(pred_name: str, gt_name: str, threshold: float = 0.5) -> bool:
    """Check if two names are similar enough."""
    pred_words = set(pred_name.lower().split())
    gt_words = set(gt_name.lower().split())
    
    if not pred_words or not gt_words:
        return False
    
    intersection = len(pred_words & gt_words)
    union = len(pred_words | gt_words)
    
    return intersection / union >= threshold


def evaluate_sample(pred_json: dict, gt_json: dict) -> dict:
    """Evaluate predicted menu against ground truth."""
    pred_items = extract_items(pred_json)
    gt_items = extract_items(gt_json)
    
    # Match items
    matched = 0
    price_correct = 0
    
    matched_gt = set()
    for pred in pred_items:
        for i, gt in enumerate(gt_items):
            if i in matched_gt:
                continue
            if fuzzy_match(pred["name"], gt["name"]):
                matched += 1
                matched_gt.add(i)
                if pred.get("price") == gt.get("price"):
                    price_correct += 1
                break
    
    precision = matched / len(pred_items) if pred_items else 0
    recall = matched / len(gt_items) if gt_items else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    price_acc = price_correct / matched if matched else 0
    
    return {
        "predicted_items": len(pred_items),
        "ground_truth_items": len(gt_items),
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "price_accuracy": price_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=Path, default=None,
                        help="Path to trained classifier model")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration")
    args = parser.parse_args()
    
    samples_dir = Path("data/samples")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Find all sample pairs
    samples = []
    for img_path in sorted(samples_dir.glob("*.jpg")):
        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            samples.append((img_path, json_path))
    
    if not samples:
        print("No samples found!")
        return
    
    print(f"Found {len(samples)} samples")
    
    # Check for GPU
    import torch
    use_gpu = args.gpu or torch.cuda.is_available()
    device = "GPU" if use_gpu else "CPU"
    print(f"Using: {device}")
    if args.model:
        print(f"Model: {args.model}")
    print()
    
    # Initialize pipeline
    config = PipelineConfig(
        use_gpu=use_gpu,
        model_path=args.model
    )
    pipeline = MenuPipeline(config)
    
    results = []
    
    for img_path, gt_path in samples:
        print(f"Processing: {img_path.name}")
        
        # Extract
        result = pipeline.process(img_path)
        pred_json = result.to_dict()
        
        # Save prediction
        pred_output = output_dir / f"{img_path.stem}_pred.json"
        with open(pred_output, 'w') as f:
            json.dump(pred_json, f, indent=2)
        
        # Save annotated image
        img_output = output_dir / f"{img_path.stem}_annotated.jpg"
        pipeline.visualize(img_path, result, img_output)
        
        # Load ground truth
        gt_json = load_ground_truth(gt_path)
        
        # Evaluate
        metrics = evaluate_sample(pred_json, gt_json)
        metrics["sample"] = img_path.name
        metrics["processing_time_ms"] = result.processing_time_ms
        results.append(metrics)
        
        print(f"  Items: {metrics['predicted_items']} pred / {metrics['ground_truth_items']} gt / {metrics['matched']} matched")
        print(f"  F1: {metrics['f1']:.2%}, Price Acc: {metrics['price_accuracy']:.2%}")
        print(f"  Time: {metrics['processing_time_ms']:.0f}ms")
        print()
    
    # Aggregate
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_price_acc = sum(r["price_accuracy"] for r in results) / len(results)
    avg_time = sum(r["processing_time_ms"] for r in results) / len(results)
    
    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Average Precision: {avg_precision:.2%}")
    print(f"Average Recall:    {avg_recall:.2%}")
    print(f"Average F1:        {avg_f1:.2%}")
    print(f"Price Accuracy:    {avg_price_acc:.2%}")
    print(f"Avg Processing:    {avg_time:.0f}ms ({device})")
    
    # Save summary
    summary = {
        "device": device,
        "model": str(args.model) if args.model else "rule-based",
        "samples": results,
        "aggregate": {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "price_accuracy": avg_price_acc,
            "avg_processing_ms": avg_time,
        }
    }
    
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/evaluation_results.json")


if __name__ == "__main__":
    main()
