"""
Ensemble classifier combining rule-based and ML approaches.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Union
from collections import Counter

from ..models.schema import OCRResult, ClassifiedText, TextElementType
from .classifier import MenuClassifier, LayoutFeatures


# Label mapping from CORD to our schema
CORD_TO_TYPE = {
    'item_name': TextElementType.ITEM_NAME,
    'item_price': TextElementType.ITEM_PRICE,
    'metadata': TextElementType.METADATA,
    'other': TextElementType.OTHER,
    'section_header': TextElementType.SECTION_HEADER,
    'group_header': TextElementType.GROUP_HEADER,
    'item_description': TextElementType.ITEM_DESCRIPTION,
}


class EnsembleClassifier:
    """
    Ensemble classifier that combines:
    1. Rule-based classification
    2. Multiple ML models
    3. Weighted voting
    """
    
    def __init__(
        self,
        model_paths: Optional[list[Path]] = None,
        use_rules: bool = True,
        rule_weight: float = 2.0,
    ):
        """
        Initialize ensemble.
        
        Parameters:
        -----------
        model_paths : List of paths to trained model pickles
        use_rules : Whether to include rule-based classifier
        rule_weight : Weight for rule-based votes (higher = more trust)
        """
        self.rule_classifier = MenuClassifier() if use_rules else None
        self.use_rules = use_rules
        self.rule_weight = rule_weight
        
        self.models = []
        self.label_encoders = []
        self.model_names = []
        
        if model_paths:
            for path in model_paths:
                self.load_model(path)
    
    def load_model(self, path: Union[str, Path]):
        """Load a trained model."""
        path = Path(path)
        if not path.exists():
            print(f"Warning: Model not found: {path}")
            return
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = data.get('model') or data.get('classifier')
        self.models.append(model)
        self.label_encoders.append(data.get('label_encoder'))
        self.model_names.append(data.get('model_type', path.stem))
        print(f"Loaded model: {self.model_names[-1]}")
    
    def _map_label(self, label_str: str) -> TextElementType:
        """Map string label to TextElementType."""
        return CORD_TO_TYPE.get(label_str.lower(), TextElementType.OTHER)
    
    def classify(
        self,
        ocr: OCRResult,
        all_ocr: list[OCRResult],
        idx: int,
        img_width: float = 1000,
        img_height: float = 1000,
    ) -> ClassifiedText:
        """Classify using ensemble voting."""
        
        votes = []
        confidences = []
        
        # Get rule-based prediction (weighted)
        if self.use_rules and self.rule_classifier:
            rule_result = self.rule_classifier.classify(
                ocr, all_ocr, idx, img_width, img_height
            )
            # Add weighted votes
            weight = int(self.rule_weight * 2)
            for _ in range(weight):
                votes.append(rule_result.label)
            confidences.append(rule_result.label_confidence)
        
        # Get ML predictions
        if self.models and self.rule_classifier:
            features = self.rule_classifier.extract_features(
                ocr, all_ocr, idx, img_width, img_height
            )
            X = features.to_array().reshape(1, -1)
            
            for model, le, name in zip(self.models, self.label_encoders, self.model_names):
                try:
                    pred_idx = model.predict(X)[0]
                    
                    # Convert to label
                    if le is not None:
                        label_str = le.inverse_transform([pred_idx])[0]
                        label = self._map_label(label_str)
                    else:
                        label = TextElementType.OTHER
                    
                    votes.append(label)
                    
                    # Get confidence
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)[0]
                        confidences.append(float(max(proba)))
                    else:
                        confidences.append(0.75)
                except Exception as e:
                    continue
        
        # Majority voting
        if votes:
            vote_counts = Counter(votes)
            winner = vote_counts.most_common(1)[0][0]
            label = winner
            confidence = np.mean(confidences) if confidences else 0.7
        else:
            label = TextElementType.ITEM_NAME
            confidence = 0.5
        
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
    
    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            'use_rules': self.use_rules,
            'rule_weight': self.rule_weight,
            'models': self.model_names,
            'n_models': len(self.models)
        }
