"""
Clean Menu OCR Pipeline.
Single-file pipeline for menu extraction with spatial-aware item-price matching.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import cv2

from src.models.schema import (
    OCRResult, ClassifiedText, MenuDocument, MenuSection, 
    MenuGroup, MenuItem, TextElementType, BoundingBox
)
from src.ocr.engine import OCREngine
from src.classifier.classifier import MenuClassifier


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    use_gpu: bool = False
    lang: str = "en"
    confidence_threshold: float = 0.2  # Lower threshold for better recall
    model_path: Optional[Path] = None


@dataclass
class PipelineResult:
    """Pipeline result with menu and visualization data."""
    document: MenuDocument
    ocr_results: list[OCRResult] = field(default_factory=list)
    classified: list[ClassifiedText] = field(default_factory=list)
    processing_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    
    def to_json(self, indent: int = 2) -> str:
        """Export to JSON."""
        return json.dumps(self.document.to_output_json(), indent=indent)
    
    def to_dict(self) -> dict:
        """Export to dictionary."""
        return self.document.to_output_json()


def slugify(text: str) -> str:
    """Convert text to slug."""
    slug = text.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    return slug.strip('_') or "unknown"


def extract_price(text: str) -> Optional[float]:
    """Extract numeric price from text."""
    cleaned = re.sub(r'[₹$£€¥]', '', text)
    cleaned = re.sub(r'(?:Rs\.?|INR)\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().replace(',', '')
    
    # Handle ranges like "500-700" -> take first
    if '-' in cleaned:
        cleaned = cleaned.split('-')[0]
    
    # Remove trailing non-digits
    cleaned = re.sub(r'[^0-9.]+$', '', cleaned)
    
    try:
        price = float(cleaned)
        if 0 < price < 100000:
            return price
    except:
        pass
    return None


def are_horizontally_aligned(bbox1: BoundingBox, bbox2: BoundingBox, tolerance: float = 1.0) -> bool:
    """Check if two boxes are on the same line with improved tolerance."""
    h1 = bbox1.height
    h2 = bbox2.height
    
    y1_mid = (bbox1.y_min + bbox1.y_max) / 2
    y2_mid = (bbox2.y_min + bbox2.y_max) / 2
    
    # Use larger tolerance for better matching
    threshold = max(h1, h2) * tolerance
    return abs(y1_mid - y2_mid) < threshold


def find_price_for_item(
    item_ct: ClassifiedText, 
    all_classified: list[ClassifiedText],
    used_prices: set[int],
    img_width: float = 1000
) -> Optional[float]:
    """Find the best price for an item using multiple strategies."""
    item_bbox = item_ct.bbox
    
    best_price = None
    best_score = float('inf')
    best_idx = -1
    
    for i, ct in enumerate(all_classified):
        if ct.label != TextElementType.ITEM_PRICE:
            continue
        if i in used_prices:
            continue
        
        price_bbox = ct.bbox
        price_val = extract_price(ct.text)
        if not price_val:
            continue
        
        # Strategy 1: Same row (horizontally aligned), price to the right
        if are_horizontally_aligned(item_bbox, price_bbox, tolerance=1.0):
            if price_bbox.x_min > item_bbox.x_min:
                # Score based on Y alignment (lower is better)
                y_diff = abs((item_bbox.y_min + item_bbox.y_max) / 2 - 
                           (price_bbox.y_min + price_bbox.y_max) / 2)
                score = y_diff
                if score < best_score:
                    best_score = score
                    best_price = price_val
                    best_idx = i
        
        # Strategy 2: Price in rightmost column (common menu layout)
        # If price is in right 30% of image and roughly aligned
        if price_bbox.x_min > img_width * 0.6:
            y_diff = abs((item_bbox.y_min + item_bbox.y_max) / 2 - 
                       (price_bbox.y_min + price_bbox.y_max) / 2)
            # Allow more vertical tolerance for column-based layouts
            if y_diff < max(item_bbox.height, price_bbox.height) * 1.5:
                score = y_diff + 10  # Slight penalty vs direct alignment
                if score < best_score:
                    best_score = score
                    best_price = price_val
                    best_idx = i
    
    if best_idx >= 0:
        used_prices.add(best_idx)
    
    return best_price


class MenuPipeline:
    """
    Clean menu extraction pipeline.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.ocr = OCREngine(
            lang=self.config.lang,
            use_gpu=self.config.use_gpu
        )
        self.classifier = MenuClassifier(
            model_path=self.config.model_path
        )
    
    def process(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> PipelineResult:
        """Process a menu image."""
        start = time.time()
        
        # Get image dimensions
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                img_w, img_h = img.size
        elif isinstance(image, Image.Image):
            img_w, img_h = image.size
        else:
            img_h, img_w = image.shape[:2]
        
        # Step 1: OCR
        ocr_results = self.ocr.extract(
            image, 
            confidence_threshold=self.config.confidence_threshold
        )
        
        if not ocr_results:
            return PipelineResult(
                document=MenuDocument(menu=[]),
                warnings=["No text detected"]
            )
        
        # Step 2: Classify
        classified = self.classifier.classify_all(
            ocr_results, 
            img_width=img_w, 
            img_height=img_h
        )
        
        # Step 3: Build structure with spatial matching
        document = self._build_menu(classified, img_w)
        
        return PipelineResult(
            document=document,
            ocr_results=ocr_results,
            classified=classified,
            processing_time_ms=(time.time() - start) * 1000
        )
    
    def _build_menu(self, classified: list[ClassifiedText], img_width: float = 1000) -> MenuDocument:
        """Build menu structure from classified text with spatial price matching."""
        sections: list[MenuSection] = []
        current_section: Optional[MenuSection] = None
        current_group: Optional[MenuGroup] = None
        used_prices: set[int] = set()
        
        for ct in classified:
            label = ct.label
            text = ct.text.strip()
            
            if label == TextElementType.SECTION_HEADER:
                # Save current
                if current_group and current_group.items:
                    if current_section:
                        current_section.groups.append(current_group)
                if current_section and current_section.groups:
                    sections.append(current_section)
                
                current_section = MenuSection(
                    id=slugify(text),
                    label=text
                )
                current_group = None
            
            elif label == TextElementType.GROUP_HEADER:
                if current_group and current_group.items:
                    if current_section:
                        current_section.groups.append(current_group)
                
                current_group = MenuGroup(
                    id=slugify(text),
                    label=text
                )
            
            elif label == TextElementType.ITEM_NAME:
                if not current_group:
                    current_group = MenuGroup(id="default", label=None)
                if not current_section:
                    current_section = MenuSection(id="menu", label="Menu")
                
                # Find price spatially
                price = find_price_for_item(ct, classified, used_prices, img_width)
                
                # Also check if price is in the text
                if not price:
                    # Look for price pattern at end of text
                    match = re.search(r'\s+([\d,]+(?:\.\d{2})?)\s*$', text)
                    if match:
                        price = extract_price(match.group(1))
                        text = text[:match.start()].strip()
                
                current_group.items.append(MenuItem(
                    name=text,
                    price=price
                ))
            
            elif label == TextElementType.ITEM_DESCRIPTION:
                if current_group and current_group.items:
                    last = current_group.items[-1]
                    if last.description is None:
                        current_group.items[-1] = MenuItem(
                            name=last.name,
                            price=last.price,
                            description=text
                        )
        
        # Save final
        if current_group and current_group.items:
            if current_section:
                current_section.groups.append(current_group)
        if current_section and current_section.groups:
            sections.append(current_section)
        
        return MenuDocument(menu=sections)
    
    def visualize(
        self,
        image_path: Union[str, Path],
        result: PipelineResult,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Visualize OCR results with bounding boxes.
        
        Returns image with bounding boxes drawn.
        """
        img = cv2.imread(str(image_path))
        
        # Color map for labels
        colors = {
            TextElementType.SECTION_HEADER: (0, 0, 255),     # Red
            TextElementType.GROUP_HEADER: (0, 165, 255),    # Orange
            TextElementType.ITEM_NAME: (0, 255, 0),         # Green
            TextElementType.ITEM_PRICE: (255, 0, 0),        # Blue
            TextElementType.ITEM_DESCRIPTION: (255, 255, 0), # Cyan
            TextElementType.METADATA: (128, 128, 128),      # Gray
            TextElementType.OTHER: (200, 200, 200),         # Light gray
        }
        
        for ct in result.classified:
            bbox = ct.bbox
            color = colors.get(ct.label, (200, 200, 200))
            
            # Draw rectangle
            pt1 = (int(bbox.x_min), int(bbox.y_min))
            pt2 = (int(bbox.x_max), int(bbox.y_max))
            cv2.rectangle(img, pt1, pt2, color, 2)
            
            # Draw label
            label_text = f"{ct.label.value[:3]}: {ct.text[:20]}"
            cv2.putText(
                img, label_text,
                (int(bbox.x_min), int(bbox.y_min) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        if output_path:
            cv2.imwrite(str(output_path), img)
        
        return img


# Convenience function
def extract_menu(
    image_path: Union[str, Path],
    output_json: Optional[Union[str, Path]] = None,
    output_image: Optional[Union[str, Path]] = None,
    use_gpu: bool = False,
) -> dict:
    """
    Extract menu from image.
    
    Parameters:
    -----------
    image_path : Path to menu image
    output_json : Optional path to save JSON
    output_image : Optional path to save visualized image
    use_gpu : Whether to use GPU
    
    Returns:
    --------
    Menu dictionary
    """
    config = PipelineConfig(use_gpu=use_gpu)
    pipeline = MenuPipeline(config)
    result = pipeline.process(image_path)
    
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            f.write(result.to_json())
    
    if output_image:
        Path(output_image).parent.mkdir(parents=True, exist_ok=True)
        pipeline.visualize(image_path, result, output_image)
    
    return result.to_dict()
