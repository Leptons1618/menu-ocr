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
        # Price patterns with Os instead of 0s
        text = re.sub(r'\bo{2,}\b', lambda m: '0' * len(m.group()), text)
        text = re.sub(r'\bO{2,}\b', lambda m: '0' * len(m.group()), text)
        
        # Mixed O/0 in prices (e.g., "1O00" -> "1000")
        text = re.sub(r'(\d)[oO](\d)', r'\g<1>0\2', text)
        text = re.sub(r'(\d)[oO](\d)', r'\g<1>0\2', text)  # Apply twice for "1OOO"
        text = re.sub(r'(\d)[oO](\d)', r'\g<1>0\2', text)  # Apply thrice
        
        # I/l as 1 in numbers
        text = re.sub(r'(\d)[Il](\d)', r'\g<1>1\2', text)
        text = re.sub(r'(\d)[Il](\d)', r'\g<1>1\2', text)
        
        # J as digit (JJooo -> 11000)
        text = re.sub(r'^[Jj]+[oO]+$', lambda m: m.group().replace('J', '1').replace('j', '1').replace('o', '0').replace('O', '0'), text)
        
        # Fix brackets in prices
        text = re.sub(r'(\d+)[}\]]', r'\1', text)
        text = re.sub(r'[{\[](\d+)', r'\1', text)
        
        # Fix "l" at start of numbers
        text = re.sub(r'^[lI](\d{2,})$', r'1\1', text)
        
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
