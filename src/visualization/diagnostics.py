"""
Diagnostic visualizations for pipeline debugging.
Generates intermediate stage visualizations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union

from ..models.schema import BoundingBox, OCRResult, ClassifiedText, TextElementType
from ..layout.columns import Column
from ..instrumentation import PipelineTrace


# Color scheme for visualization
COLORS = {
    TextElementType.SECTION_HEADER: (0, 0, 255),      # Red
    TextElementType.GROUP_HEADER: (0, 165, 255),     # Orange
    TextElementType.ITEM_NAME: (0, 255, 0),          # Green
    TextElementType.ITEM_PRICE: (255, 0, 0),         # Blue
    TextElementType.ITEM_DESCRIPTION: (255, 255, 0), # Cyan
    TextElementType.METADATA: (128, 128, 128),       # Gray
    TextElementType.OTHER: (200, 200, 200),          # Light gray
}

COLUMN_COLORS = [
    (255, 100, 100),  # Light blue
    (100, 255, 100),  # Light green
    (100, 100, 255),  # Light red
    (255, 255, 100),  # Light cyan
    (255, 100, 255),  # Light magenta
    (100, 255, 255),  # Light yellow
]


class DiagnosticVisualizer:
    """Generate diagnostic visualizations at each pipeline stage."""
    
    def __init__(self, font_scale: float = 0.4, line_thickness: int = 2):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        font_scale : Font scale for text labels
        line_thickness : Line thickness for boxes
        """
        self.font_scale = font_scale
        self.line_thickness = line_thickness
    
    def visualize_detections(
        self,
        image_path: Union[str, Path],
        boxes: list[BoundingBox],
        confidences: Optional[list[float]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Draw all detection boxes (including low-confidence).
        
        Parameters:
        -----------
        image_path : Path to source image
        boxes : List of bounding boxes
        confidences : Optional confidence scores for coloring
        output_path : Optional path to save visualization
        
        Returns:
        --------
        Annotated image as numpy array
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        for i, box in enumerate(boxes):
            # Color based on confidence
            if confidences and i < len(confidences):
                conf = confidences[i]
                if conf > 0.7:
                    color = (0, 255, 0)  # Green - high confidence
                elif conf > 0.4:
                    color = (0, 165, 255)  # Orange - medium
                else:
                    color = (0, 0, 255)  # Red - low confidence
            else:
                color = (0, 255, 0)
            
            pt1 = (int(box.x_min), int(box.y_min))
            pt2 = (int(box.x_max), int(box.y_max))
            cv2.rectangle(img, pt1, pt2, color, self.line_thickness)
            
            # Add confidence label
            if confidences and i < len(confidences):
                label = f"{confidences[i]:.2f}"
                cv2.putText(
                    img, label,
                    (int(box.x_min), int(box.y_min) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 1
                )
        
        if output_path:
            cv2.imwrite(str(output_path), img)
        
        return img
    
    def visualize_columns(
        self,
        image_path: Union[str, Path],
        columns: list[Column],
        boxes: Optional[list[BoundingBox]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Draw detected column boundaries.
        
        Parameters:
        -----------
        image_path : Path to source image
        columns : Detected columns
        boxes : Optional boxes to show column assignment
        output_path : Optional path to save visualization
        
        Returns:
        --------
        Annotated image as numpy array
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height = img.shape[0]
        
        for i, col in enumerate(columns):
            color = COLUMN_COLORS[i % len(COLUMN_COLORS)]
            
            # Draw column boundaries
            cv2.line(img, (int(col.x_min), 0), (int(col.x_min), height), color, 2)
            cv2.line(img, (int(col.x_max), 0), (int(col.x_max), height), (color[0]//2, color[1]//2, color[2]//2), 1)
            
            # Label column
            label = f"Col {i}"
            if col.is_price_column:
                label += " (PRICE)"
            cv2.putText(
                img, label,
                (int(col.x_min) + 5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            
            # Fill column area with semi-transparent color
            overlay = img.copy()
            cv2.rectangle(
                overlay,
                (int(col.x_min), 0),
                (int(col.x_max), height),
                color, -1
            )
            img = cv2.addWeighted(overlay, 0.1, img, 0.9, 0)
        
        # Draw boxes colored by column assignment
        if boxes:
            for i, box in enumerate(boxes):
                # Find which column this box belongs to
                box_center = (box.x_min + box.x_max) / 2
                col_idx = 0
                for j, col in enumerate(columns):
                    if col.x_min <= box_center <= col.x_max:
                        col_idx = j
                        break
                
                color = COLUMN_COLORS[col_idx % len(COLUMN_COLORS)]
                pt1 = (int(box.x_min), int(box.y_min))
                pt2 = (int(box.x_max), int(box.y_max))
                cv2.rectangle(img, pt1, pt2, color, 1)
        
        if output_path:
            cv2.imwrite(str(output_path), img)
        
        return img
    
    def visualize_classifications(
        self,
        image_path: Union[str, Path],
        classified: list[ClassifiedText],
        output_path: Optional[Union[str, Path]] = None,
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        Color-coded classification results.
        
        Parameters:
        -----------
        image_path : Path to source image
        classified : List of classified text elements
        output_path : Optional path to save visualization
        show_labels : Whether to show text labels
        
        Returns:
        --------
        Annotated image as numpy array
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        for ct in classified:
            bbox = ct.bbox
            color = COLORS.get(ct.label, (200, 200, 200))
            
            pt1 = (int(bbox.x_min), int(bbox.y_min))
            pt2 = (int(bbox.x_max), int(bbox.y_max))
            cv2.rectangle(img, pt1, pt2, color, self.line_thickness)
            
            if show_labels:
                # Short label
                label_short = ct.label.value[:3].upper()
                text_preview = ct.text[:15] + "..." if len(ct.text) > 15 else ct.text
                label = f"{label_short}: {text_preview}"
                
                cv2.putText(
                    img, label,
                    (int(bbox.x_min), int(bbox.y_min) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 1
                )
        
        # Add legend
        legend_y = 30
        for label_type, color in COLORS.items():
            cv2.rectangle(img, (10, legend_y - 15), (25, legend_y), color, -1)
            cv2.putText(
                img, label_type.value,
                (30, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
            legend_y += 20
        
        if output_path:
            cv2.imwrite(str(output_path), img)
        
        return img
    
    def visualize_matches(
        self,
        image_path: Union[str, Path],
        items: list[ClassifiedText],
        prices: list[ClassifiedText],
        matches: dict[int, int],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Draw lines connecting matched item-price pairs.
        
        Parameters:
        -----------
        image_path : Path to source image
        items : List of item ClassifiedText
        prices : List of price ClassifiedText
        matches : Dict mapping item indices to price indices
        output_path : Optional path to save visualization
        
        Returns:
        --------
        Annotated image as numpy array
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Draw all items (green)
        for i, item in enumerate(items):
            bbox = item.bbox
            color = (0, 255, 0) if i in matches else (100, 100, 100)
            pt1 = (int(bbox.x_min), int(bbox.y_min))
            pt2 = (int(bbox.x_max), int(bbox.y_max))
            cv2.rectangle(img, pt1, pt2, color, 2)
        
        # Draw all prices (blue)
        matched_prices = set(matches.values())
        for j, price in enumerate(prices):
            bbox = price.bbox
            color = (255, 0, 0) if j in matched_prices else (100, 100, 100)
            pt1 = (int(bbox.x_min), int(bbox.y_min))
            pt2 = (int(bbox.x_max), int(bbox.y_max))
            cv2.rectangle(img, pt1, pt2, color, 2)
        
        # Draw match lines
        for item_idx, price_idx in matches.items():
            if item_idx >= len(items) or price_idx >= len(prices):
                continue
            
            item = items[item_idx]
            price = prices[price_idx]
            
            # Line from item right edge to price left edge
            item_center = (
                int(item.bbox.x_max),
                int((item.bbox.y_min + item.bbox.y_max) / 2)
            )
            price_center = (
                int(price.bbox.x_min),
                int((price.bbox.y_min + price.bbox.y_max) / 2)
            )
            
            cv2.line(img, item_center, price_center, (0, 200, 200), 2)
            
            # Draw arrow head
            cv2.circle(img, price_center, 5, (0, 200, 200), -1)
        
        # Add stats
        total_items = len(items)
        total_prices = len(prices)
        total_matches = len(matches)
        stats = f"Items: {total_items}, Prices: {total_prices}, Matches: {total_matches}"
        cv2.putText(
            img, stats,
            (10, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        if output_path:
            cv2.imwrite(str(output_path), img)
        
        return img
    
    def visualize_hierarchy(
        self,
        image_path: Union[str, Path],
        classified: list[ClassifiedText],
        font_levels: Optional[list[int]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Visualize hierarchy with font level annotations.
        
        Parameters:
        -----------
        image_path : Path to source image
        classified : Classified text elements
        font_levels : Font level for each element (0=largest)
        output_path : Optional path to save
        
        Returns:
        --------
        Annotated image
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        for i, ct in enumerate(classified):
            bbox = ct.bbox
            color = COLORS.get(ct.label, (200, 200, 200))
            
            # Draw box
            pt1 = (int(bbox.x_min), int(bbox.y_min))
            pt2 = (int(bbox.x_max), int(bbox.y_max))
            cv2.rectangle(img, pt1, pt2, color, 2)
            
            # Add font level if available
            if font_levels and i < len(font_levels):
                level = font_levels[i]
                level_label = f"L{level}"
                cv2.putText(
                    img, level_label,
                    (int(bbox.x_max) + 5, int(bbox.y_min) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
                )
        
        if output_path:
            cv2.imwrite(str(output_path), img)
        
        return img
    
    def generate_full_diagnostic(
        self,
        image_path: Union[str, Path],
        trace: PipelineTrace,
        output_dir: Union[str, Path],
        classified: Optional[list[ClassifiedText]] = None,
        columns: Optional[list[Column]] = None,
        items: Optional[list[ClassifiedText]] = None,
        prices: Optional[list[ClassifiedText]] = None,
        matches: Optional[dict[int, int]] = None,
    ) -> dict[str, Path]:
        """
        Generate complete diagnostic report with all visualizations.
        
        Returns:
        --------
        Dict mapping visualization name to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        image_path = Path(image_path)
        base_name = image_path.stem
        
        # 1. Detection visualization
        if trace.detections:
            boxes = [
                BoundingBox(**d) for d in trace.detections
            ]
            det_path = output_dir / f"{base_name}_1_detections.jpg"
            self.visualize_detections(image_path, boxes, output_path=det_path)
            outputs['detections'] = det_path
        
        # 2. Column visualization
        if columns:
            boxes = [BoundingBox(**d) for d in trace.detections] if trace.detections else None
            col_path = output_dir / f"{base_name}_2_columns.jpg"
            self.visualize_columns(image_path, columns, boxes, output_path=col_path)
            outputs['columns'] = col_path
        
        # 3. Classification visualization
        if classified:
            cls_path = output_dir / f"{base_name}_3_classifications.jpg"
            self.visualize_classifications(image_path, classified, output_path=cls_path)
            outputs['classifications'] = cls_path
        
        # 4. Matching visualization
        if items and prices and matches is not None:
            match_path = output_dir / f"{base_name}_4_matches.jpg"
            self.visualize_matches(image_path, items, prices, matches, output_path=match_path)
            outputs['matches'] = match_path
        
        # 5. Save trace JSON
        trace_path = output_dir / f"{base_name}_trace.json"
        trace.save(output_dir)
        outputs['trace'] = trace_path
        
        # 6. Save summary text
        summary_path = output_dir / f"{base_name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(trace.get_summary())
        outputs['summary'] = summary_path
        
        return outputs
