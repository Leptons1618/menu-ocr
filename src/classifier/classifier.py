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
    
    # Enhanced price patterns (more comprehensive)
    PRICE_PATTERNS = [
        r'^[\$£€₹¥]?\s*\d+(?:[.,]\d{1,2})?\s*$',
        r'^\d{2,5}$',  # Just digits (common in menus)
        r'^[\$£€₹¥]\s*\d+(?:[.,]\d{1,2})?',  # $12, ₹500, €15.50
        r'^\d+(?:[.,]\d{1,2})?\s*[\$£€₹¥]$',  # 12$, 500₹
        r'^(?:Rs\.?|INR|USD|EUR|GBP)\s*\d+(?:[.,]\d{1,2})?',  # Rs. 500
        r'^\d+(?:[.,]\d{2})\s*$',  # 12.99, 500.00
        r'^(?:MRP|Price)[:\s]*[\$£€₹¥]?\s*\d+',  # MRP: 500
        r'^\d+\s*(?:/-|/-\s*$)',  # 500/- (Indian style)
        r'^\d+\s*/\s*\d+',  # 500/700 (price range)
        r'^(?:from|starting)\s*[\$£€₹¥]?\s*\d+',  # from $10
        r'^\+?\s*[\$£€₹¥]?\s*\d+',  # +$5 (addon)
    ]
    
    # Section keywords (expanded)
    SECTION_WORDS = {
        'appetizers', 'appetiser', 'starters', 'starter', 'mains', 'main course',
        'entrees', 'entrées', 'desserts', 'dessert', 'beverages', 'beverage',
        'drinks', 'drink', 'wines', 'wine', 'cocktails', 'cocktail', 'sides', 'side',
        'breakfast', 'lunch', 'dinner', 'brunch', 'specials', 'special', 
        'soups', 'soup', 'salads', 'salad', 'seafood', 'meat', 'meats',
        'vegetarian', 'vegan', 'menu', 'carte', 'today', 'chef',
        'tandoor', 'tandoori', 'biryani', 'biryanis', 'rice', 'rices',
        'bread', 'breads', 'naan', 'roti', 'rotis', 'curries', 'curry',
        'noodles', 'noodle', 'pasta', 'pastas', 'pizza', 'pizzas',
        'burgers', 'burger', 'sandwiches', 'sandwich', 'wraps', 'wrap',
        'sushi', 'sashimi', 'rolls', 'roll', 'combo', 'combos', 'platter', 'platters',
        'snacks', 'snack', 'finger food', 'finger foods', 'mezze', 'tapas',
        'hot', 'cold', 'fresh', 'signature', 'house', 'daily', 'seasonal',
        'mocktails', 'mocktail', 'juices', 'juice', 'smoothies', 'smoothie',
        'shakes', 'shake', 'coffee', 'tea', 'lemonade', 'refreshers',
        'spirits', 'liquor', 'liquors', 'whiskey', 'whisky', 'vodka', 'rum',
        'gin', 'tequila', 'brandy', 'cognac', 'beer', 'beers', 'lager', 'ale',
        'wine list', 'by the glass', 'by the bottle',
    }
    
    # Group/category keywords (expanded)
    CATEGORY_WORDS = {
        'deluxe', 'premium', 'special', 'imported', 'domestic',
        'single', 'double', 'malts', 'malt', 'bourbon', 'tennessee', 
        'beer', 'wine', 'rum', 'brandy', 'liqueur', 'cocktail', 
        'bottle', 'pint', 'glass', 'pitcher', 'carafe', 'jug',
        'strong', 'mild', 'light', 'dark', 'medium', 
        'classic', 'house', 'signature', 'featured', 'recommended',
        'kids', 'children', 'family', 'sharing', 'for two',
        'vegetarian', 'vegan', 'gluten-free', 'dairy-free', 
        'organic', 'fresh', 'homemade', 'hand-crafted',
        'grilled', 'fried', 'steamed', 'baked', 'roasted', 'smoked',
        'hot', 'cold', 'iced', 'frozen', 'chilled',
        'small', 'medium', 'large', 'regular', 'extra', 
        'half', 'full', 'quarter', 'whole',
        'add-ons', 'extras', 'toppings', 'sides',
        'indian', 'chinese', 'italian', 'mexican', 'thai', 'japanese',
        'american', 'continental', 'mediterranean', 'asian', 'european',
        'red', 'white', 'rosé', 'sparkling', 'champagne', 'prosecco',
        'draft', 'draught', 'bottled', 'canned', 'tap',
        'scotch', 'irish', 'japanese', 'american', 'canadian',
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
        
        # Compile regex patterns for speed
        self._price_patterns = [re.compile(p, re.IGNORECASE) for p in self.PRICE_PATTERNS]
        
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
    
    def _is_price_text(self, text: str) -> bool:
        """Check if text matches any price pattern."""
        text_clean = text.strip()
        return any(p.match(text_clean) for p in self._price_patterns)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching."""
        # Common OCR error corrections
        text = text.replace('|', 'l')  # pipe to l
        text = re.sub(r'[oO](?=\d)', '0', text)  # O before digit -> 0
        text = re.sub(r'(?<=\d)[oO]', '0', text)  # O after digit -> 0
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
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
        text = self._preprocess_text(ocr.text)
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
        
        # Price detection using compiled patterns
        is_price = self._is_price_text(text)
        is_price_only = is_price and len(text.strip()) <= 12
        
        # Category detection (check individual words)
        text_lower = text.lower()
        text_words = set(re.findall(r'\b\w+\b', text_lower))
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
        text = self._preprocess_text(text)
        text_lower = text.lower().strip()
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Apply lexical priors if available
        prior_boost = {}
        if lexical_priors:
            prior_boost = lexical_priors
        
        # PRICE DETECTION (highest priority)
        # Check price column position first
        if features.is_price_column and features.digit_ratio > 0.3:
            return TextElementType.ITEM_PRICE, 0.95
        
        # Price-only text
        if features.is_price_only:
            return TextElementType.ITEM_PRICE, 0.95
        
        # High digit ratio with price pattern
        if features.has_price and features.digit_ratio > 0.4:
            return TextElementType.ITEM_PRICE, 0.90
        
        # Right-aligned numeric text (common price position)
        if features.rel_x > 0.7 and features.digit_ratio > 0.5 and features.word_count <= 2:
            return TextElementType.ITEM_PRICE, 0.85
        
        # SECTION HEADER DETECTION
        is_section_keyword = bool(text_words & self.SECTION_WORDS)
        section_conf = 0.0
        
        # Strong section signals
        if is_section_keyword and features.word_count <= 4:
            section_conf = 0.90
        elif features.font_level == 0 and features.word_count <= 4 and not features.has_price:
            section_conf = 0.85
        elif (features.is_all_caps and 
              features.rel_height > 1.2 and 
              features.gap_below > 1.2 and
              features.word_count <= 5 and
              not features.has_price):
            section_conf = 0.80
        # Centered text with large font
        elif (0.3 < features.rel_x < 0.7 and
              features.font_level <= 1 and
              features.word_count <= 4 and
              not features.has_price):
            section_conf = 0.75
        
        section_conf += prior_boost.get(TextElementType.SECTION_HEADER, 0.0)
        if section_conf >= 0.75:
            return TextElementType.SECTION_HEADER, min(section_conf, 0.95)
        
        # GROUP HEADER DETECTION
        group_conf = 0.0
        is_group_keyword = bool(text_words & self.CATEGORY_WORDS)
        
        if is_group_keyword and features.word_count <= 4 and not features.has_price:
            group_conf = 0.85
        elif features.font_level == 1 and features.word_count <= 4 and not features.has_price:
            group_conf = 0.75
        elif (features.has_category_word and 
              features.word_count <= 3 and 
              not features.has_price):
            group_conf = 0.80
        
        group_conf += prior_boost.get(TextElementType.GROUP_HEADER, 0.0)
        if group_conf >= 0.70:
            return TextElementType.GROUP_HEADER, min(group_conf, 0.90)
        
        # DESCRIPTION DETECTION
        # Long text, not all caps, smaller font, multiple words
        if (features.word_count > 6 and 
            not features.is_all_caps and
            features.font_level >= 2):
            desc_conf = 0.80 + prior_boost.get(TextElementType.ITEM_DESCRIPTION, 0.0)
            return TextElementType.ITEM_DESCRIPTION, min(desc_conf, 0.85)
        
        # Text starting with common description phrases
        desc_starters = ['served', 'comes', 'includes', 'topped', 'made', 'with', 'our', 'a ']
        if any(text_lower.startswith(s) for s in desc_starters) and features.word_count > 3:
            desc_conf = 0.75 + prior_boost.get(TextElementType.ITEM_DESCRIPTION, 0.0)
            return TextElementType.ITEM_DESCRIPTION, min(desc_conf, 0.85)
        
        # FILTER OCR NOISE - reject obviously bad text
        # Too short or too many unusual characters
        if len(text) < 3:
            return TextElementType.OTHER, 0.80
        
        # Single word with mixed case pattern like "RuX", "jOn" (OCR noise)
        if features.word_count == 1 and len(text) <= 4:
            # Check for noise patterns
            has_mixed_inner = any(c.isupper() for c in text[1:-1]) if len(text) > 2 else False
            if has_mixed_inner:
                return TextElementType.OTHER, 0.85
        
        # All digits or mostly digits without price pattern (OCR noise like "1J00")
        if features.digit_ratio > 0.6 and not features.has_price:
            return TextElementType.OTHER, 0.80
        
        # Very short text with high upper ratio but not all caps
        if len(text) <= 4 and features.upper_ratio > 0.5 and not text.isupper():
            return TextElementType.OTHER, 0.75
        
        # Very short single words that aren't common food abbreviations
        VALID_SHORT_WORDS = {
            'tea', 'dal', 'dip', 'pie', 'ham', 'jam', 'egg', 'bun', 'nut', 'pop',
            'ice', 'ale', 'rum', 'gin', 'veg', 'non', 'hot', 'dry', 'mix', 'raw',
            'bbq', 'sub', 'fry', 'soy', 'cod', 'red', 'pav', 'naan', 'rice', 'soup',
            'beef', 'pork', 'lamb', 'fish', 'crab', 'tuna', 'milk', 'cola', 'soda',
            'wine', 'beer', 'coke', 'chai', 'roti', 'dosa', 'idli', 'upma', 'poha',
        }
        if features.word_count == 1 and len(text) <= 5:
            if text_lower not in VALID_SHORT_WORDS:
                # Not a recognized short food word - likely noise
                return TextElementType.OTHER, 0.70
        
        # ITEM NAME DETECTION (default for remaining text)
        # Be stricter: require reasonable text patterns
        if features.word_count >= 1 and features.digit_ratio < 0.3:
            # Additional checks for valid item names
            # Must have at least one alphabetic character
            if not any(c.isalpha() for c in text):
                return TextElementType.OTHER, 0.80
            # Must have reasonable letter ratio
            alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
            if alpha_ratio < 0.5:
                return TextElementType.OTHER, 0.75
            
            name_conf = 0.75 + prior_boost.get(TextElementType.ITEM_NAME, 0.0)
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
