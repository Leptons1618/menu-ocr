"""
Unified OCR Engine using EasyOCR.
"""

import re
from pathlib import Path
from typing import Union, Optional
import numpy as np
from PIL import Image
import easyocr

from ..models.schema import BoundingBox, OCRResult


class OCREngine:
    """
    EasyOCR-based text extraction engine.
    """
    
    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self.reader = easyocr.Reader(
            [lang],
            gpu=use_gpu,
            verbose=False
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean OCR text - fix common errors."""
        # Fix common OCR errors
        replacements = [
            # Price-like patterns that got corrupted
            (r'\bo{2,}\b', lambda m: '0' * len(m.group())),  # ooo -> 000
            (r'\bO{2,}\b', lambda m: '0' * len(m.group())),  # OOO -> 000
            (r'\bl{2,}\b', lambda m: '1' * len(m.group())),  # lll -> 111
            (r'\bI{2,}\b', lambda m: '1' * len(m.group())),  # III -> 111
        ]
        
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        
        return text.strip()
    
    def extract(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        confidence_threshold: float = 0.3,
    ) -> list[OCRResult]:
        """
        Extract text from image.
        
        Parameters:
        -----------
        image : Image source
        confidence_threshold : Minimum confidence for results
        
        Returns:
        --------
        List of OCRResult with text, bounding box, and confidence
        """
        # Convert to path or array
        if isinstance(image, Path):
            image = str(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Run OCR
        raw_results = self.reader.readtext(image)
        
        # Convert to OCRResult
        results = []
        for bbox_pts, text, conf in raw_results:
            if conf < confidence_threshold:
                continue
            
            # Clean text
            text = self._clean_text(text)
            if not text:
                continue
            
            # Convert polygon to axis-aligned bbox
            pts = np.array(bbox_pts)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            
            results.append(OCRResult(
                text=text,
                bbox=BoundingBox(
                    x_min=float(x_min),
                    y_min=float(y_min),
                    x_max=float(x_max),
                    y_max=float(y_max),
                ),
                confidence=float(conf),
            ))
        
        # Sort by position (top to bottom, left to right)
        results.sort(key=lambda r: (r.bbox.y_min, r.bbox.x_min))
        
        return results
