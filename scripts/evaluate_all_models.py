#!/usr/bin/env python3
"""Evaluate all trained models on sample menu images."""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import MenuPipeline, PipelineConfig
from src.classifier.classifier import MenuClassifier


@dataclass
class EvaluationResult:
    """Evaluation metrics for a model."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    price_accuracy: float
    avg_time_ms: float
    total_items_pred: int
    total_items_gt: int
    correct_items: int


def load_ground_truth(json_path: Path) -> dict:
    """Load ground truth JSON."""
    with open(json_path) as f:
        return json.load(f)


def extract_items_from_result(result: dict) -> list[tuple[str, Optional[float]]]:
    """Extract (name, price) pairs from result."""
    items = []
    menu = result.get("menu", [])
    for section in menu:
        for group in section.get("groups", []):
            for item in group.get("items", []):
                name = item.get("name", "").lower().strip()
                price = item.get("price")
                if name:
                    items.append((name, price))
    return items


def calculate_metrics(pred_items: list, gt_items: list) -> dict:
    """Calculate precision, recall, F1, and price accuracy."""
    pred_names = set(name for name, _ in pred_items)
    gt_names = set(name for name, _ in gt_items)
    
    # Name matching
    correct_names = pred_names & gt_names
    precision = len(correct_names) / len(pred_names) if pred_names else 0
    recall = len(correct_names) / len(gt_names) if gt_names else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Price accuracy for matched items
    pred_dict = {name: price for name, price in pred_items}
    gt_dict = {name: price for name, price in gt_items}
    
    price_correct = 0
    price_total = 0
    for name in correct_names:
        pred_price = pred_dict.get(name)
        gt_price = gt_dict.get(name)
        if gt_price is not None:
            price_total += 1
            if pred_price == gt_price:
                price_correct += 1
    
    price_accuracy = price_correct / price_total if price_total > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "price_accuracy": price_accuracy,
        "correct_items": len(correct_names),
    }


def evaluate_model(model_path: Optional[Path], model_name: str, sample_dir: Path) -> EvaluationResult:
    """Evaluate a single model on all samples."""
    
    # Configure pipeline
    config = PipelineConfig(
        use_gpu=True,
        model_path=model_path,
    )
    
    pipeline = MenuPipeline(config)
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_price_acc = 0
    total_time = 0
    total_pred = 0
    total_gt = 0
    total_correct = 0
    num_samples = 0
    
    # Find all sample images
    for img_path in sorted(sample_dir.glob("*.jpg")):
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue
        
        # Load ground truth
        gt_data = load_ground_truth(json_path)
        gt_items = extract_items_from_result(gt_data)
        
        # Run inference
        start = time.time()
        result = pipeline.process(str(img_path))
        elapsed = (time.time() - start) * 1000
        
        # Extract predicted items
        pred_items = extract_items_from_result(result.to_dict())
        
        # Calculate metrics
        metrics = calculate_metrics(pred_items, gt_items)
        
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_f1 += metrics["f1_score"]
        total_price_acc += metrics["price_accuracy"]
        total_time += elapsed
        total_pred += len(pred_items)
        total_gt += len(gt_items)
        total_correct += metrics["correct_items"]
        num_samples += 1
    
    if num_samples == 0:
        raise ValueError("No samples found")
    
    return EvaluationResult(
        model_name=model_name,
        accuracy=total_correct / total_gt if total_gt > 0 else 0,
        precision=total_precision / num_samples,
        recall=total_recall / num_samples,
        f1_score=total_f1 / num_samples,
        price_accuracy=total_price_acc / num_samples,
        avg_time_ms=total_time / num_samples,
        total_items_pred=total_pred,
        total_items_gt=total_gt,
        correct_items=total_correct,
    )


def evaluate_ensemble(sample_dir: Path, models_dir: Path) -> EvaluationResult:
    """Evaluate ensemble approach on samples."""
    from src.classifier.ensemble import EnsembleClassifier
    from src.ocr.engine import OCREngine
    from PIL import Image
    
    # Load all ML models
    model_paths = [
        models_dir / "rf_classifier.pkl",
        models_dir / "xgb_classifier.pkl",
    ]
    model_paths = [p for p in model_paths if p.exists()]
    
    ensemble = EnsembleClassifier(
        model_paths=model_paths,
        use_rules=True,
        rule_weight=2.0,
    )
    
    ocr = OCREngine(use_gpu=True)
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_price_acc = 0
    total_time = 0
    total_pred = 0
    total_gt = 0
    total_correct = 0
    num_samples = 0
    
    for img_path in sorted(sample_dir.glob("*.jpg")):
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue
        
        gt_data = load_ground_truth(json_path)
        gt_items = extract_items_from_result(gt_data)
        
        # Get image dimensions
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        start = time.time()
        
        # OCR
        ocr_results = ocr.extract(str(img_path))
        
        # Ensemble classify
        classified = ensemble.classify_all(ocr_results, img_w, img_h)
        
        # Build menu (reuse pipeline logic)
        from src.pipeline import MenuPipeline, PipelineConfig
        pipeline = MenuPipeline(PipelineConfig(use_gpu=True))
        document = pipeline._build_menu(classified)
        
        elapsed = (time.time() - start) * 1000
        
        pred_items = extract_items_from_result(document.to_output_json())
        metrics = calculate_metrics(pred_items, gt_items)
        
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_f1 += metrics["f1_score"]
        total_price_acc += metrics["price_accuracy"]
        total_time += elapsed
        total_pred += len(pred_items)
        total_gt += len(gt_items)
        total_correct += metrics["correct_items"]
        num_samples += 1
    
    if num_samples == 0:
        raise ValueError("No samples found")
    
    return EvaluationResult(
        model_name="Ensemble (Rule+ML)",
        accuracy=total_correct / total_gt if total_gt > 0 else 0,
        precision=total_precision / num_samples,
        recall=total_recall / num_samples,
        f1_score=total_f1 / num_samples,
        price_accuracy=total_price_acc / num_samples,
        avg_time_ms=total_time / num_samples,
        total_items_pred=total_pred,
        total_items_gt=total_gt,
        correct_items=total_correct,
    )


def main():
    """Run evaluation on all models."""
    sample_dir = Path("data/samples")
    models_dir = Path("models")
    
    # Define models to evaluate
    models = [
        (None, "Rule-Based"),
        (models_dir / "rf_classifier.pkl", "Random Forest"),
        (models_dir / "gb_classifier.pkl", "Gradient Boosting"),
        (models_dir / "xgb_classifier.pkl", "XGBoost"),
        (models_dir / "mlp_classifier.pkl", "MLP Neural Net"),
    ]
    
    results = []
    
    print("=" * 80)
    print("MODEL EVALUATION ON SAMPLE MENUS")
    print("=" * 80)
    print()
    
    for model_path, model_name in models:
        if model_path and not model_path.exists():
            print(f"[SKIP] {model_name}: Model not found")
            continue
        
        print(f"Evaluating: {model_name}...")
        try:
            result = evaluate_model(model_path, model_name, sample_dir)
            results.append(result)
            print(f"  F1: {result.f1_score:.1%}, Price Acc: {result.price_accuracy:.1%}, Time: {result.avg_time_ms:.0f}ms")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Evaluate ensemble
    print("Evaluating: Ensemble (Rule + ML)...")
    try:
        ensemble_result = evaluate_ensemble(sample_dir, models_dir)
        results.append(ensemble_result)
        print(f"  F1: {ensemble_result.f1_score:.1%}, Price Acc: {ensemble_result.price_accuracy:.1%}, Time: {ensemble_result.avg_time_ms:.0f}ms")
    except Exception as e:
        print(f"  Error: {e}")
    
    print()
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    
    # Print table header
    print(f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Price Acc':>12} {'Time (ms)':>12}")
    print("-" * 74)
    
    for r in results:
        print(f"{r.model_name:<20} {r.precision:>10.1%} {r.recall:>10.1%} {r.f1_score:>10.1%} {r.price_accuracy:>12.1%} {r.avg_time_ms:>12.0f}")
    
    print()
    
    # Find best model
    if results:
        best = max(results, key=lambda x: x.f1_score)
        print(f"Best model by F1: {best.model_name} ({best.f1_score:.1%})")
    
    # Save results to JSON
    output = {
        "results": [
            {
                "model": r.model_name,
                "precision": round(r.precision, 4),
                "recall": round(r.recall, 4),
                "f1_score": round(r.f1_score, 4),
                "price_accuracy": round(r.price_accuracy, 4),
                "avg_time_ms": round(r.avg_time_ms, 1),
                "items_predicted": r.total_items_pred,
                "items_ground_truth": r.total_items_gt,
                "correct_items": r.correct_items,
            }
            for r in results
        ]
    }
    
    output_path = models_dir / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
