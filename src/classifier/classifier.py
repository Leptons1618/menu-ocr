"""
Enhanced text classifier for menu elements.
Supports rule-based, ML-based, and hybrid approaches.
ML models serve as secondary signals, not primary decision makers.
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
    
    # Extended features for hierarchy
    font_level: int = 2  # 0=largest, higher=smaller
    column_index: int = 0
    is_price_column: bool = False
    
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


class HybridClassifier:
    """
    Hybrid classifier: rule-based primary, ML as secondary signal.
    
    ML models trained on receipt datasets underperform on menus due to
    domain mismatch. This classifier uses rules as the primary decision
    maker and only consults ML for low-confidence cases.
    """
    
    # Price patterns
    PRICE_PATTERNS = [
        r'^[\$£€₹¥]?\s*\d+(?:[.,]\d{1,2})?\s*$',
        r'^\d{2,5}$',  # Just digits (common in menus)
        r'^(?:Rs\.?|INR)\s*\d+',
    ]
    
    # Section keywords
    SECTION_WORDS = {
        'appetizers', 'starters', 'mains', 'entrees', 'desserts',
        'beverages', 'drinks', 'wines', 'cocktails', 'sides',
        'breakfast', 'lunch', 'dinner', 'specials', 'soups', 'salads',
        'seafood', 'meat', 'vegetarian', 'menu',
    }
    
    # Group/category keywords
    CATEGORY_WORDS = {
        'deluxe', 'premium', 'special', 'imported', 'domestic',
        'single', 'malts', 'bourbon', 'tennessee', 'beer', 'wine',
        'rum', 'brandy', 'liqueur', 'cocktail', 'bottle', 'pint',
        'strong', 'mild', 'appetizer', 'starter', 'main', 'dessert',
        'beverage', 'drink', 'spirit', 'whiskey', 'vodka', 'gin',
        'classic', 'house', 'signature', 'kids', 'vegetarian', 'vegan',
    }
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        ml_confidence_threshold: float = 0.7,
        rule_override_threshold: float = 0.85,
    ):
        """
        Initialize hybrid classifier.
        
        Parameters:
        -----------
        model_path : Path to trained ML model (optional)
        ml_confidence_threshold : Min ML confidence to consider its prediction
        rule_override_threshold : Rule confidence above which ML is ignored
        """
        self.model = None
        self.label_encoder = None
        self.is_trained = False
        
        self.ml_confidence_threshold = ml_confidence_threshold
        self.rule_override_threshold = rule_override_threshold
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def _map_cord_label(self, label_str: str) -> TextElementType:
        """Map CORD/training labels to TextElementType."""
        mapping = {
            'item_name': TextElementType.ITEM_NAME,
            'item_price': TextElementType.ITEM_PRICE,
            'metadata': TextElementType.METADATA,
            'other': TextElementType.OTHER,
            'section_header': TextElementType.SECTION_HEADER,
            'group_header': TextElementType.GROUP_HEADER,
            'item_description': TextElementType.ITEM_DESCRIPTION,
        }
        return mapping.get(label_str.lower(), TextElementType.OTHER)
    
    def extract_features(
        self,
        ocr: OCRResult,
        all_ocr: list[OCRResult],
        idx: int,
        img_width: float = 1000,
        img_height: float = 1000,
        font_level: int = 2,
        column_index: int = 0,
        is_price_column: bool = False,
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
        text_words = set(text_lower.split())
        has_cat = bool(text_words & self.CATEGORY_WORDS)
        
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
            font_level=font_level,
            column_index=column_index,
            is_price_column=is_price_column,
        )
    
    def classify_rule_based(
        self,
        features: LayoutFeatures,
        text: str,
        lexical_priors: Optional[dict] = None,
    ) -> tuple[TextElementType, float]:
        """
        Rule-based classification with lexical priors.
        
        Returns (label, confidence) tuple.
        """
        text_lower = text.lower().strip()
        text_words = set(text_lower.split())
        
        # Apply lexical priors if available
        prior_boost = {}
        if lexical_priors:
            prior_boost = lexical_priors
        
        # Price detection (highest priority)
        if features.is_price_only or (features.is_price_column and features.digit_ratio > 0.4):
            return TextElementType.ITEM_PRICE, 0.95
        
        if features.has_price and features.digit_ratio > 0.5:
            return TextElementType.ITEM_PRICE, 0.90
        
        # Section header detection
        # Large font (level 0-1), possibly all caps, section keywords
        is_section_keyword = bool(text_words & self.SECTION_WORDS)
        section_conf = 0.0
        
        if is_section_keyword and features.word_count <= 4:
            section_conf = 0.85
        elif features.font_level == 0 and features.word_count <= 4:
            section_conf = 0.80
        elif (features.is_all_caps and 
              features.rel_height > 1.3 and 
              features.gap_below > 1.5 and
              features.word_count <= 4):
            section_conf = 0.75
        
        section_conf += prior_boost.get(TextElementType.SECTION_HEADER, 0.0)
        if section_conf >= 0.75:
            return TextElementType.SECTION_HEADER, min(section_conf, 0.95)
        
        # Group header detection
        # Medium font (level 1), category keywords
        group_conf = 0.0
        if features.has_category_word and features.word_count <= 3:
            group_conf = 0.80
        elif features.font_level == 1 and features.word_count <= 4:
            group_conf = 0.70
        
        group_conf += prior_boost.get(TextElementType.GROUP_HEADER, 0.0)
        if group_conf >= 0.70:
            return TextElementType.GROUP_HEADER, min(group_conf, 0.90)
        
        # Description: long text, not caps, smaller font
        if features.word_count > 6 and not features.is_all_caps:
            desc_conf = 0.75 + prior_boost.get(TextElementType.ITEM_DESCRIPTION, 0.0)
            return TextElementType.ITEM_DESCRIPTION, min(desc_conf, 0.85)
        
        # Item name: default for regular text
        if features.word_count >= 1 and features.digit_ratio < 0.5:
            name_conf = 0.70 + prior_boost.get(TextElementType.ITEM_NAME, 0.0)
            return TextElementType.ITEM_NAME, min(name_conf, 0.85)
        
        return TextElementType.OTHER, 0.50
    
    def classify_hybrid(
        self,
        features: LayoutFeatures,
        text: str,
        lexical_priors: Optional[dict] = None,
    ) -> tuple[TextElementType, float]:
        """
        Hybrid classification: rule-based primary, ML secondary.
        
        ML is only consulted when rule confidence is below threshold.
        """
        # Get rule-based prediction
        rule_label, rule_conf = self.classify_rule_based(features, text, lexical_priors)
        
        # If rule confidence is high, use rule prediction
        if rule_conf >= self.rule_override_threshold:
            return rule_label, rule_conf
        
        # If no ML model, use rule prediction
        if not self.is_trained or self.model is None:
            return rule_label, rule_conf
        
        # Get ML prediction
        X = features.to_array().reshape(1, -1)
        try:
            pred_idx = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            ml_conf = float(max(proba))
            
            if self.label_encoder is not None:
                label_str = self.label_encoder.inverse_transform([pred_idx])[0]
                ml_label = self._map_cord_label(label_str)
            else:
                ml_label = TextElementType.OTHER
        except Exception:
            return rule_label, rule_conf
        
        # Decision logic:
        # 1. If ML agrees with rules, boost confidence
        # 2. If ML strongly disagrees AND rule confidence is low, consider ML
        # 3. Otherwise, trust rules
        
        if ml_label == rule_label:
            # Agreement: boost confidence
            combined_conf = min(0.95, (rule_conf + ml_conf) / 2 + 0.1)
            return rule_label, combined_conf
        
        if rule_conf < 0.6 and ml_conf > self.ml_confidence_threshold:
            # Low rule confidence, high ML confidence: consider ML
            # But discount ML due to domain mismatch
            discounted_ml_conf = ml_conf * 0.7
            if discounted_ml_conf > rule_conf:
                return ml_label, discounted_ml_conf
        
        # Default: trust rules
        return rule_label, rule_conf
    
    def classify(
        self,
        ocr: OCRResult,
        all_ocr: list[OCRResult],
        idx: int,
        img_width: float = 1000,
        img_height: float = 1000,
        font_level: int = 2,
        column_index: int = 0,
        is_price_column: bool = False,
        lexical_priors: Optional[dict] = None,
    ) -> ClassifiedText:
        """Classify a single OCR result using hybrid approach."""
        features = self.extract_features(
            ocr, all_ocr, idx, img_width, img_height,
            font_level, column_index, is_price_column
        )
        
        label, confidence = self.classify_hybrid(features, ocr.text, lexical_priors)
        
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
        font_levels: Optional[list[int]] = None,
        column_indices: Optional[list[int]] = None,
        price_column_idx: Optional[int] = None,
        lexical_priors_list: Optional[list[dict]] = None,
    ) -> list[ClassifiedText]:
        """Classify all OCR results with optional enrichment data."""
        results = []
        
        for i, ocr in enumerate(ocr_results):
            font_level = font_levels[i] if font_levels and i < len(font_levels) else 2
            col_idx = column_indices[i] if column_indices and i < len(column_indices) else 0
            is_price_col = (price_column_idx is not None and col_idx == price_column_idx)
            lexical = lexical_priors_list[i] if lexical_priors_list and i < len(lexical_priors_list) else None
            
            result = self.classify(
                ocr, ocr_results, i, img_width, img_height,
                font_level, col_idx, is_price_col, lexical
            )
            results.append(result)
        
        return results
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        """Train the ML component (used as secondary signal only)."""
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
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
            }, f)
    
    def load(self, path: Path):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data.get('model') or data.get('classifier')
        self.label_encoder = data.get('label_encoder')
        self.is_trained = True


# Backward compatibility alias
MenuClassifier = HybridClassifier
