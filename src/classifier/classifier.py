"""
Enhanced text classifier for menu elements.
Supports rule-based, ML-based, and hybrid approaches.
"""

import re
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..models.schema import OCRResult, ClassifiedText, TextElementType, BoundingBox


@dataclass
class LayoutFeatures:
    """Layout and content features for classification."""
    # Position (normalized 0-1)
    rel_x: float
    rel_y: float
    
    # Size (relative to average)
    rel_height: float
    rel_width: float
    
    # Gaps
    gap_above: float
    gap_below: float
    
    # Content
    text_length: int
    word_count: int
    digit_ratio: float
    upper_ratio: float
    
    # Patterns
    is_all_caps: bool
    is_title_case: bool
    has_price: bool
    is_price_only: bool
    has_category_word: bool
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML."""
        return np.array([
            self.rel_x, self.rel_y, self.rel_height, self.rel_width,
            self.gap_above, self.gap_below, self.text_length, self.word_count,
            self.digit_ratio, self.upper_ratio,
            float(self.is_all_caps), float(self.is_title_case),
            float(self.has_price), float(self.is_price_only),
            float(self.has_category_word)
        ])


class MenuClassifier:
    """
    Classifier for menu text elements.
    """
    
    # Price patterns
    PRICE_PATTERNS = [
        r'^[\$£€₹]?\s*\d+(?:[.,]\d{1,2})?\s*$',
        r'^\d{2,5}$',  # Just digits (common in menus)
    ]
    
    # Category keywords
    CATEGORY_WORDS = {
        'deluxe', 'premium', 'special', 'imported', 'domestic',
        'single', 'malts', 'bourbon', 'tennessee', 'beer', 'wine',
        'rum', 'brandy', 'liqueur', 'cocktail', 'bottle', 'pint',
        'strong', 'mild', 'appetizer', 'starter', 'main', 'dessert',
        'beverage', 'drink', 'spirit', 'whiskey', 'vodka', 'gin',
    }
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize classifier."""
        self.model = None
        self.is_trained = False
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def extract_features(
        self,
        ocr: OCRResult,
        all_ocr: list[OCRResult],
        idx: int,
        img_width: float = 1000,
        img_height: float = 1000,
    ) -> LayoutFeatures:
        """Extract features for a single OCR result."""
        text = ocr.text
        bbox = ocr.bbox
        
        # Compute averages
        heights = [o.bbox.height for o in all_ocr]
        widths = [o.bbox.width for o in all_ocr]
        avg_h = np.mean(heights) if heights else bbox.height
        avg_w = np.mean(widths) if widths else bbox.width
        
        # Gaps
        gap_above = 0.0
        gap_below = 0.0
        if idx > 0:
            prev = all_ocr[idx - 1]
            gap_above = max(0, bbox.y_min - prev.bbox.y_max)
        if idx < len(all_ocr) - 1:
            next_o = all_ocr[idx + 1]
            gap_below = max(0, next_o.bbox.y_min - bbox.y_max)
        
        # Content analysis
        digit_count = sum(1 for c in text if c.isdigit())
        upper_count = sum(1 for c in text if c.isupper())
        
        # Price detection
        is_price = any(re.match(p, text.strip()) for p in self.PRICE_PATTERNS)
        is_price_only = is_price and len(text.strip()) <= 10
        
        # Category detection
        text_lower = text.lower()
        has_cat = any(w in text_lower for w in self.CATEGORY_WORDS)
        
        return LayoutFeatures(
            rel_x=bbox.x_min / img_width,
            rel_y=bbox.y_min / img_height,
            rel_height=bbox.height / avg_h if avg_h > 0 else 1.0,
            rel_width=bbox.width / avg_w if avg_w > 0 else 1.0,
            gap_above=gap_above / avg_h if avg_h > 0 else 0.0,
            gap_below=gap_below / avg_h if avg_h > 0 else 0.0,
            text_length=len(text),
            word_count=len(text.split()),
            digit_ratio=digit_count / max(len(text), 1),
            upper_ratio=upper_count / max(len(text), 1),
            is_all_caps=text.isupper() and len(text) > 2,
            is_title_case=text.istitle(),
            has_price=is_price,
            is_price_only=is_price_only,
            has_category_word=has_cat,
        )
    
    def classify_rule_based(self, features: LayoutFeatures, text: str) -> tuple[TextElementType, float]:
        """Rule-based classification."""
        
        # Price detection (highest priority)
        if features.is_price_only:
            return TextElementType.ITEM_PRICE, 0.95
        
        if features.has_price and features.digit_ratio > 0.5:
            return TextElementType.ITEM_PRICE, 0.90
        
        # Section header: large, caps, big gap below
        if (features.is_all_caps and 
            features.rel_height > 1.3 and 
            features.gap_below > 1.5 and
            features.word_count <= 4):
            return TextElementType.SECTION_HEADER, 0.85
        
        # Group header: category word
        if features.has_category_word and features.word_count <= 3:
            return TextElementType.GROUP_HEADER, 0.80
        
        # Description: long text, not caps
        if features.word_count > 6 and not features.is_all_caps:
            return TextElementType.ITEM_DESCRIPTION, 0.80
        
        # Item name: default for text
        if features.word_count >= 1 and features.digit_ratio < 0.5:
            return TextElementType.ITEM_NAME, 0.70
        
        return TextElementType.OTHER, 0.50
    
    def classify(
        self,
        ocr: OCRResult,
        all_ocr: list[OCRResult],
        idx: int,
        img_width: float = 1000,
        img_height: float = 1000,
    ) -> ClassifiedText:
        """Classify a single OCR result."""
        features = self.extract_features(ocr, all_ocr, idx, img_width, img_height)
        
        if self.is_trained and self.model is not None:
            # Use trained model
            X = features.to_array().reshape(1, -1)
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            label = TextElementType(pred)
            confidence = float(max(proba))
        else:
            # Use rule-based
            label, confidence = self.classify_rule_based(features, ocr.text)
        
        return ClassifiedText(
            ocr=ocr,
            label=label,
            label_confidence=confidence
        )
    
    def classify_all(
        self,
        ocr_results: list[OCRResult],
        img_width: float = 1000,
        img_height: float = 1000,
    ) -> list[ClassifiedText]:
        """Classify all OCR results."""
        return [
            self.classify(ocr, ocr_results, i, img_width, img_height)
            for i, ocr in enumerate(ocr_results)
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        """Train the classifier."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred)
        }
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model}, f)
    
    def load(self, path: Path):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.is_trained = True
