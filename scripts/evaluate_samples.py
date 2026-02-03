#!/usr/bin/env python3
"""
Comprehensive evaluation script for Menu OCR.
Tests all samples against ground truth and reports detailed metrics.
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from difflib import SequenceMatcher

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MenuPipeline, PipelineConfig


@dataclass
class ItemMatch:
    """Represents a matched/unmatched item."""
    gt_name: Optional[str] = None
    gt_price: Optional[float] = None
    pred_name: Optional[str] = None
    pred_price: Optional[float] = None
    name_similarity: float = 0.0
    is_matched: bool = False
    price_correct: bool = False


@dataclass
class EvalResult:
    """Evaluation result for a single image."""
    filename: str
    gt_items: int = 0
    pred_items: int = 0
    matched_items: int = 0
    gt_prices: int = 0
    pred_prices: int = 0
    correct_prices: int = 0
    gt_sections: int = 0
    pred_sections: int = 0
    processing_time_ms: float = 0.0
    matches: list = field(default_factory=list)
    
    @property
    def precision(self) -> float:
        return self.matched_items / max(self.pred_items, 1)
    
    @property
    def recall(self) -> float:
        return self.matched_items / max(self.gt_items, 1)
    
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 0.001)
    
    @property
    def price_accuracy(self) -> float:
        return self.correct_prices / max(self.gt_prices, 1)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def text_similarity(a: str, b: str) -> float:
    """Calculate text similarity using SequenceMatcher."""
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def extract_items(menu_data: dict) -> list[tuple[str, Optional[float]]]:
    """Extract all items from menu structure."""
    items = []
    for section in menu_data.get('menu', []):
        for group in section.get('groups', []):
            for item in group.get('items', []):
                name = item.get('name', '')
                price = item.get('price')
                if name:
                    items.append((name, price))
    return items


def match_items(gt_items: list, pred_items: list, threshold: float = 0.6) -> list[ItemMatch]:
    """Match predicted items to ground truth using text similarity."""
    matches = []
    used_pred = set()
    
    for gt_name, gt_price in gt_items:
        best_match = None
        best_sim = 0.0
        best_idx = -1
        
        for i, (pred_name, pred_price) in enumerate(pred_items):
            if i in used_pred:
                continue
            sim = text_similarity(gt_name, pred_name)
            if sim > best_sim:
                best_sim = sim
                best_match = (pred_name, pred_price)
                best_idx = i
        
        if best_sim >= threshold and best_idx >= 0:
            used_pred.add(best_idx)
            price_correct = (gt_price is not None and 
                          best_match[1] is not None and 
                          abs(gt_price - best_match[1]) < 1)
            matches.append(ItemMatch(
                gt_name=gt_name,
                gt_price=gt_price,
                pred_name=best_match[0],
                pred_price=best_match[1],
                name_similarity=best_sim,
                is_matched=True,
                price_correct=price_correct
            ))
        else:
            matches.append(ItemMatch(
                gt_name=gt_name,
                gt_price=gt_price,
                is_matched=False
            ))
    
    # Add unmatched predictions
    for i, (pred_name, pred_price) in enumerate(pred_items):
        if i not in used_pred:
            matches.append(ItemMatch(
                pred_name=pred_name,
                pred_price=pred_price,
                is_matched=False
            ))
    
    return matches


def evaluate_single(
    pipeline: MenuPipeline,
    image_path: Path,
    gt_path: Path
) -> EvalResult:
    """Evaluate a single image against ground truth."""
    # Load ground truth
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    # Process image
    result = pipeline.process(str(image_path))
    pred_data = result.to_dict()
    
    # Extract items
    gt_items = extract_items(gt_data)
    pred_items = extract_items(pred_data)
    
    # Match items
    matches = match_items(gt_items, pred_items)
    
    # Calculate metrics
    matched = [m for m in matches if m.is_matched]
    gt_with_price = [m for m in matches if m.gt_price is not None and m.is_matched]
    correct_prices = [m for m in gt_with_price if m.price_correct]
    
    return EvalResult(
        filename=image_path.name,
        gt_items=len(gt_items),
        pred_items=len(pred_items),
        matched_items=len(matched),
        gt_prices=len([1 for n, p in gt_items if p is not None]),
        pred_prices=len([1 for n, p in pred_items if p is not None]),
        correct_prices=len(correct_prices),
        gt_sections=len(gt_data.get('menu', [])),
        pred_sections=len(pred_data.get('menu', [])),
        processing_time_ms=result.processing_time_ms,
        matches=matches
    )


def run_evaluation(samples_dir: Path, use_gpu: bool = True) -> list[EvalResult]:
    """Run evaluation on all samples."""
    config = PipelineConfig(use_gpu=use_gpu)
    pipeline = MenuPipeline(config)
    
    results = []
    image_files = sorted(samples_dir.glob('*.jpg'))
    
    print(f"\nEvaluating {len(image_files)} images...")
    print("=" * 80)
    
    for img_path in image_files:
        gt_path = img_path.with_suffix('.json')
        if not gt_path.exists():
            print(f"Skipping {img_path.name} - no ground truth")
            continue
        
        try:
            result = evaluate_single(pipeline, img_path, gt_path)
            results.append(result)
            
            print(f"{result.filename:20s} | "
                  f"P:{result.precision:.2f} R:{result.recall:.2f} F1:{result.f1:.2f} | "
                  f"Price:{result.price_accuracy:.2f} | "
                  f"Items: {result.matched_items}/{result.gt_items} | "
                  f"{result.processing_time_ms:.0f}ms")
        except Exception as e:
            print(f"{img_path.name:20s} | ERROR: {e}")
    
    return results


def print_summary(results: list[EvalResult]):
    """Print evaluation summary."""
    if not results:
        print("No results to summarize")
        return
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    total_gt = sum(r.gt_items for r in results)
    total_pred = sum(r.pred_items for r in results)
    total_matched = sum(r.matched_items for r in results)
    total_gt_prices = sum(r.gt_prices for r in results)
    total_correct_prices = sum(r.correct_prices for r in results)
    total_time = sum(r.processing_time_ms for r in results)
    
    precision = total_matched / max(total_pred, 1)
    recall = total_matched / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    price_acc = total_correct_prices / max(total_gt_prices, 1)
    
    print(f"\nImages Evaluated: {len(results)}")
    print(f"Total Ground Truth Items: {total_gt}")
    print(f"Total Predicted Items: {total_pred}")
    print(f"Total Matched Items: {total_matched}")
    print(f"\n--- Metrics ---")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")
    print(f"F1 Score: {f1:.1%}")
    print(f"Price Accuracy: {price_acc:.1%} ({total_correct_prices}/{total_gt_prices})")
    print(f"\n--- Performance ---")
    print(f"Total Time: {total_time/1000:.1f}s")
    print(f"Avg Time per Image: {total_time/len(results):.0f}ms")
    
    # Per-image breakdown
    print(f"\n--- Per-Image F1 Distribution ---")
    f1_scores = sorted([r.f1 for r in results])
    print(f"Min: {min(f1_scores):.2f}, Max: {max(f1_scores):.2f}, Median: {f1_scores[len(f1_scores)//2]:.2f}")
    
    # Find worst performing images
    print(f"\n--- Worst Performing Images ---")
    worst = sorted(results, key=lambda r: r.f1)[:5]
    for r in worst:
        print(f"  {r.filename}: F1={r.f1:.2f}, Matched={r.matched_items}/{r.gt_items}")
    
    # Find best performing images
    print(f"\n--- Best Performing Images ---")
    best = sorted(results, key=lambda r: r.f1, reverse=True)[:5]
    for r in best:
        print(f"  {r.filename}: F1={r.f1:.2f}, Matched={r.matched_items}/{r.gt_items}")


def analyze_errors(results: list[EvalResult]):
    """Analyze common error patterns."""
    print(f"\n--- Error Analysis ---")
    
    # Collect unmatched items
    missed_gt = []
    false_positives = []
    price_errors = []
    
    for r in results:
        for m in r.matches:
            if m.gt_name and not m.is_matched:
                missed_gt.append((r.filename, m.gt_name, m.gt_price))
            elif m.pred_name and not m.is_matched:
                false_positives.append((r.filename, m.pred_name, m.pred_price))
            elif m.is_matched and m.gt_price and not m.price_correct:
                price_errors.append((r.filename, m.gt_name, m.gt_price, m.pred_price))
    
    print(f"\nMissed Ground Truth Items: {len(missed_gt)}")
    if missed_gt[:5]:
        for fn, name, price in missed_gt[:5]:
            print(f"  [{fn}] {name} (price: {price})")
    
    print(f"\nFalse Positive Items: {len(false_positives)}")
    if false_positives[:5]:
        for fn, name, price in false_positives[:5]:
            print(f"  [{fn}] {name} (price: {price})")
    
    print(f"\nPrice Errors: {len(price_errors)}")
    if price_errors[:5]:
        for fn, name, gt_p, pred_p in price_errors[:5]:
            print(f"  [{fn}] {name}: GT={gt_p}, Pred={pred_p}")


if __name__ == "__main__":
    samples_dir = Path("data/samples")
    
    if not samples_dir.exists():
        print(f"Samples directory not found: {samples_dir}")
        sys.exit(1)
    
    results = run_evaluation(samples_dir, use_gpu=True)
    print_summary(results)
    analyze_errors(results)
