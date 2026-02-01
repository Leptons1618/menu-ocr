#!/usr/bin/env python3
"""
Train classifier on CORD-v2 with multiple algorithms.
Supports: Random Forest, Gradient Boosting, XGBoost, MLP
"""

import sys
import json
import argparse
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.schema import TextElementType, BoundingBox, OCRResult
from src.classifier.classifier import MenuClassifier, LayoutFeatures


# CORD category mapping
CORD_LABEL_MAP = {
    'menu.nm': TextElementType.ITEM_NAME,
    'menu.price': TextElementType.ITEM_PRICE,
    'menu.cnt': TextElementType.METADATA,
    'menu.sub_nm': TextElementType.ITEM_DESCRIPTION,
    'menu.sub_price': TextElementType.ITEM_PRICE,
    'menu.etc': TextElementType.OTHER,
    'sub_total.subtotal_price': TextElementType.ITEM_PRICE,
    'sub_total.service_price': TextElementType.ITEM_PRICE,
    'sub_total.tax_price': TextElementType.ITEM_PRICE,
    'total.total_price': TextElementType.ITEM_PRICE,
    'total.total_etc': TextElementType.METADATA,
}


def extract_cord_features(max_samples: int = 5000):
    """Extract features from CORD dataset."""
    from datasets import load_from_disk, load_dataset
    
    print("Loading CORD-v2 dataset...")
    
    cord_path = Path("data/cord")
    if cord_path.exists():
        dataset = load_from_disk(str(cord_path))
        train_data = dataset['train']
    else:
        dataset = load_dataset("naver-clova-ix/cord-v2", split="train")
        train_data = dataset
    
    X_list = []
    y_list = []
    classifier = MenuClassifier()
    
    print(f"Processing {min(max_samples, len(train_data))} samples...")
    
    for idx in tqdm(range(min(max_samples, len(train_data)))):
        try:
            sample = train_data[idx]
            gt = json.loads(sample['ground_truth'])
            
            img_size = gt.get('meta', {}).get('image_size', {})
            img_w = img_size.get('width', 1000)
            img_h = img_size.get('height', 1000)
            
            valid_lines = gt.get('valid_line', [])
            
            ocr_results = []
            labels = []
            
            for line in valid_lines:
                category = line.get('category', 'other')
                words = line.get('words', [])
                
                if not words:
                    continue
                
                text_parts = []
                all_x = []
                all_y = []
                
                for word in words:
                    text_parts.append(word.get('text', ''))
                    quad = word.get('quad', {})
                    all_x.extend([quad.get('x1', 0), quad.get('x2', 0), 
                                  quad.get('x3', 0), quad.get('x4', 0)])
                    all_y.extend([quad.get('y1', 0), quad.get('y2', 0),
                                  quad.get('y3', 0), quad.get('y4', 0)])
                
                if not text_parts or not all_x:
                    continue
                
                text = ' '.join(text_parts)
                bbox = BoundingBox(
                    x_min=min(all_x),
                    y_min=min(all_y),
                    x_max=max(all_x),
                    y_max=max(all_y)
                )
                
                ocr_results.append(OCRResult(
                    text=text,
                    bbox=bbox,
                    confidence=1.0
                ))
                
                label = CORD_LABEL_MAP.get(category, TextElementType.OTHER)
                labels.append(label.value)
            
            for i, (ocr, label) in enumerate(zip(ocr_results, labels)):
                features = classifier.extract_features(
                    ocr, ocr_results, i, img_w, img_h
                )
                X_list.append(features.to_array())
                y_list.append(label)
        
        except Exception as e:
            continue
    
    return np.array(X_list), np.array(y_list)


def get_model(model_type: str):
    """Get model instance by type."""
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    elif model_type == "gradient_boost":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=150,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
    
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    
    elif model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(X: np.ndarray, y: np.ndarray, model_type: str = "random_forest"):
    """Train classifier."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    
    print(f"\nTraining {model_type} classifier...")
    print(f"Dataset: {len(X)} samples")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")
    
    # Encode labels for XGBoost
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train
    model = get_model(model_type)
    
    import time
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Convert back for report
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)
    report = classification_report(y_test_labels, y_pred_labels)
    
    print(f"\nTraining time: {train_time:.1f}s")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return model, le, {
        'model_type': model_type,
        'accuracy': accuracy,
        'train_time': train_time,
        'report': report,
        'n_samples': len(X),
        'n_classes': len(unique)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-samples", type=int, default=5000,
        help="Maximum training samples"
    )
    parser.add_argument(
        "--model", type=str, default="random_forest",
        choices=["random_forest", "gradient_boost", "xgboost", "mlp"],
        help="Model type"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("models/classifier.pkl"),
        help="Output model path"
    )
    args = parser.parse_args()
    
    # Extract features
    X, y = extract_cord_features(args.max_samples)
    
    if len(X) == 0:
        print("No training data found!")
        return
    
    # Train
    model, label_encoder, results = train(X, y, args.model)
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump({
            'model': model,
            'label_encoder': label_encoder,
            'model_type': args.model,
            'results': results
        }, f)
    
    print(f"\nModel saved to: {args.output}")
    
    # Save results to JSON for paper
    results_path = args.output.with_suffix('.json')
    with open(results_path, 'w') as f:
        json.dump({
            'model_type': args.model,
            'accuracy': results['accuracy'],
            'train_time': results['train_time'],
            'n_samples': results['n_samples']
        }, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
