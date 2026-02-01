"""
Clean Menu OCR Pipeline.
Integrates column-aware layout, bipartite matching, hierarchy detection,
and hybrid classification for improved menu extraction.
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
from src.classifier.classifier import HybridClassifier
from src.layout.columns import ColumnDetector, Column
from src.layout.reading_order import ReadingOrderResolver
from src.matching.bipartite import PriceItemMatcher
from src.hierarchy.font_analysis import FontScaleAnalyzer
from src.hierarchy.lexical import LexicalPriors
from src.hierarchy.fsm import HierarchyFSM
from src.instrumentation import PipelineInstrumentor, StageTimer


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    use_gpu: bool = False
    lang: str = "en"
    confidence_threshold: float = 0.2  # Lower threshold for better recall
    model_path: Optional[Path] = None
    
    # New configuration options
    use_column_detection: bool = True
    use_bipartite_matching: bool = True
    use_hierarchy_fsm: bool = True
    use_lexical_priors: bool = True
    save_diagnostics: bool = False
    diagnostics_dir: Optional[Path] = None


@dataclass
class PipelineResult:
    """Pipeline result with menu and visualization data."""
    document: MenuDocument
    ocr_results: list[OCRResult] = field(default_factory=list)
    classified: list[ClassifiedText] = field(default_factory=list)
    processing_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    
    # Extended results
    columns: list[Column] = field(default_factory=list)
    matches: dict = field(default_factory=dict)
    trace: Optional[object] = None
    
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


class MenuPipeline:
    """
    Enhanced menu extraction pipeline with:
    - Column-aware layout segmentation
    - Bipartite matching for price-item association
    - Hierarchy detection with FSM constraints
    - Hybrid classification (rules + ML)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Core components
        self.ocr = OCREngine(
            lang=self.config.lang,
            use_gpu=self.config.use_gpu
        )
        self.classifier = HybridClassifier(
            model_path=self.config.model_path
        )
        
        # Layout components
        self.column_detector = ColumnDetector() if self.config.use_column_detection else None
        self.reading_order = ReadingOrderResolver()
        
        # Matching components
        self.matcher = PriceItemMatcher() if self.config.use_bipartite_matching else None
        
        # Hierarchy components
        self.font_analyzer = FontScaleAnalyzer()
        self.lexical = LexicalPriors() if self.config.use_lexical_priors else None
        self.fsm = HierarchyFSM() if self.config.use_hierarchy_fsm else None
    
    def process(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> PipelineResult:
        """Process a menu image with full pipeline."""
        start = time.time()
        
        # Initialize instrumentation
        image_path = str(image) if isinstance(image, (str, Path)) else "memory"
        instrumentor = PipelineInstrumentor(image_path, vars(self.config))
        
        # Get image dimensions
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                img_w, img_h = img.size
        elif isinstance(image, Image.Image):
            img_w, img_h = image.size
        else:
            img_h, img_w = image.shape[:2]
        
        # Stage 1: OCR
        instrumentor.start_stage('ocr_detection')
        ocr_results = self.ocr.extract(
            image, 
            confidence_threshold=self.config.confidence_threshold
        )
        instrumentor.end_stage('ocr_detection')
        instrumentor.record_ocr_results(ocr_results)
        
        if not ocr_results:
            trace = instrumentor.finalize()
            return PipelineResult(
                document=MenuDocument(menu=[]),
                warnings=["No text detected"],
                trace=trace
            )
        
        # Stage 2: Column Detection
        columns = []
        price_col_idx = None
        column_indices = [0] * len(ocr_results)
        
        if self.column_detector:
            instrumentor.start_stage('column_detection')
            boxes = [r.bbox for r in ocr_results]
            texts = [r.text for r in ocr_results]
            
            columns = self.column_detector.detect_columns(boxes, img_w)
            price_col_idx = self.column_detector.detect_price_column(columns, boxes, texts)
            
            # Assign each box to a column
            for i, box in enumerate(boxes):
                column_indices[i] = self.column_detector.assign_box_to_column(box, columns)
            
            instrumentor.end_stage('column_detection')
            instrumentor.record_columns(columns)
        
        # Stage 3: Font Analysis
        instrumentor.start_stage('classification')
        boxes = [r.bbox for r in ocr_results]
        font_levels = self.font_analyzer.analyze(boxes)
        
        # Stage 4: Lexical Priors
        lexical_priors = None
        if self.lexical:
            lexical_priors = [self.lexical.compute_prior(r.text) for r in ocr_results]
        
        # Stage 5: Classification with enriched features
        classified = self.classifier.classify_all(
            ocr_results,
            img_width=img_w,
            img_height=img_h,
            font_levels=font_levels,
            column_indices=column_indices,
            price_column_idx=price_col_idx,
            lexical_priors_list=lexical_priors
        )
        instrumentor.end_stage('classification')
        instrumentor.record_classifications(classified)
        
        # Stage 6: Hierarchy FSM (optional sequence correction)
        if self.fsm:
            instrumentor.start_stage('hierarchy_decoding')
            # Build observation scores from classifications
            observations = []
            for ct in classified:
                obs = {t: -5.0 for t in TextElementType}
                obs[ct.label] = np.log(max(ct.label_confidence, 0.01))
                observations.append(obs)
            
            # Decode with Viterbi
            corrected_labels = self.fsm.viterbi_decode(observations)
            
            # Update classifications with corrected labels
            for i, new_label in enumerate(corrected_labels):
                if new_label != classified[i].label:
                    # Create new ClassifiedText with corrected label
                    classified[i] = ClassifiedText(
                        ocr=classified[i].ocr,
                        label=new_label,
                        label_confidence=classified[i].label_confidence * 0.9  # Slight penalty
                    )
            instrumentor.end_stage('hierarchy_decoding')
        
        # Stage 7: Bipartite Matching
        matches = {}
        if self.matcher:
            instrumentor.start_stage('matching')
            items = [c for c in classified if c.label == TextElementType.ITEM_NAME]
            prices = [c for c in classified if c.label == TextElementType.ITEM_PRICE]
            
            # Get item/price indices in original list
            item_indices = [i for i, c in enumerate(classified) if c.label == TextElementType.ITEM_NAME]
            price_indices = [i for i, c in enumerate(classified) if c.label == TextElementType.ITEM_PRICE]
            
            if items and prices:
                local_matches = self.matcher.match(items, prices, columns, img_w)
                # Map back to original indices
                for local_item_idx, local_price_idx in local_matches.items():
                    orig_item_idx = item_indices[local_item_idx]
                    orig_price_idx = price_indices[local_price_idx]
                    matches[orig_item_idx] = orig_price_idx
            
            instrumentor.end_stage('matching')
            instrumentor.record_matches(matches, items, prices)
        
        # Stage 8: Build menu structure
        instrumentor.start_stage('structure_building')
        document = self._build_menu_with_matches(classified, matches, columns, img_w)
        instrumentor.end_stage('structure_building')
        
        trace = instrumentor.finalize()
        
        return PipelineResult(
            document=document,
            ocr_results=ocr_results,
            classified=classified,
            processing_time_ms=(time.time() - start) * 1000,
            columns=columns,
            matches=matches,
            trace=trace
        )
    
    def _build_menu_with_matches(
        self,
        classified: list[ClassifiedText],
        matches: dict[int, int],
        columns: list[Column],
        img_width: float = 1000
    ) -> MenuDocument:
        """Build menu structure using bipartite matching results."""
        sections: list[MenuSection] = []
        current_section: Optional[MenuSection] = None
        current_group: Optional[MenuGroup] = None
        
        for i, ct in enumerate(classified):
            label = ct.label
            text = ct.text.strip()
            
            if label == TextElementType.SECTION_HEADER:
                # Save current group/section
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
                
                # Get price from bipartite matching
                price = None
                if i in matches:
                    price_idx = matches[i]
                    if price_idx < len(classified):
                        price = extract_price(classified[price_idx].text)
                
                # Fallback: check if price embedded in text
                if price is None:
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
        
        # Save final group/section
        if current_group and current_group.items:
            if current_section:
                current_section.groups.append(current_group)
        if current_section and current_section.groups:
            sections.append(current_section)
        
        return MenuDocument(menu=sections)
    
    def _build_menu(self, classified: list[ClassifiedText], img_width: float = 1000) -> MenuDocument:
        """Legacy menu building without bipartite matching."""
        return self._build_menu_with_matches(classified, {}, [], img_width)
    
    def visualize(
        self,
        image_path: Union[str, Path],
        result: PipelineResult,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Visualize OCR results with bounding boxes."""
        img = cv2.imread(str(image_path))
        
        colors = {
            TextElementType.SECTION_HEADER: (0, 0, 255),
            TextElementType.GROUP_HEADER: (0, 165, 255),
            TextElementType.ITEM_NAME: (0, 255, 0),
            TextElementType.ITEM_PRICE: (255, 0, 0),
            TextElementType.ITEM_DESCRIPTION: (255, 255, 0),
            TextElementType.METADATA: (128, 128, 128),
            TextElementType.OTHER: (200, 200, 200),
        }
        
        for ct in result.classified:
            bbox = ct.bbox
            color = colors.get(ct.label, (200, 200, 200))
            
            pt1 = (int(bbox.x_min), int(bbox.y_min))
            pt2 = (int(bbox.x_max), int(bbox.y_max))
            cv2.rectangle(img, pt1, pt2, color, 2)
            
            label_text = f"{ct.label.value[:3]}: {ct.text[:20]}"
            cv2.putText(
                img, label_text,
                (int(bbox.x_min), int(bbox.y_min) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        # Draw match lines
        if result.matches:
            items = [c for c in result.classified if c.label == TextElementType.ITEM_NAME]
            prices = [c for c in result.classified if c.label == TextElementType.ITEM_PRICE]
            item_indices = [i for i, c in enumerate(result.classified) if c.label == TextElementType.ITEM_NAME]
            price_indices = [i for i, c in enumerate(result.classified) if c.label == TextElementType.ITEM_PRICE]
            
            for orig_item_idx, orig_price_idx in result.matches.items():
                if orig_item_idx < len(result.classified) and orig_price_idx < len(result.classified):
                    item = result.classified[orig_item_idx]
                    price = result.classified[orig_price_idx]
                    
                    item_center = (int(item.bbox.x_max), int((item.bbox.y_min + item.bbox.y_max) / 2))
                    price_center = (int(price.bbox.x_min), int((price.bbox.y_min + price.bbox.y_max) / 2))
                    cv2.line(img, item_center, price_center, (0, 200, 200), 1)
        
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
