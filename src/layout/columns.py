"""
Column detection via x-coordinate clustering.
Identifies document columns and price column for menu layouts.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.cluster import DBSCAN

from ..models.schema import BoundingBox, OCRResult


@dataclass
class Column:
    """Represents a detected column in the document."""
    index: int
    x_min: float
    x_max: float
    box_indices: list[int] = field(default_factory=list)
    is_price_column: bool = False
    
    @property
    def center(self) -> float:
        return (self.x_min + self.x_max) / 2
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min


class ColumnDetector:
    """Detect document columns via x-coordinate clustering."""
    
    def __init__(
        self,
        min_column_gap: float = 30.0,
        min_boxes_per_column: int = 3,
        price_column_threshold: float = 0.7,
    ):
        """
        Initialize column detector.
        
        Parameters:
        -----------
        min_column_gap : Minimum horizontal gap between columns (pixels)
        min_boxes_per_column : Minimum boxes to form a column
        price_column_threshold : Minimum digit ratio to consider price column
        """
        self.min_column_gap = min_column_gap
        self.min_boxes_per_column = min_boxes_per_column
        self.price_column_threshold = price_column_threshold
    
    def detect_columns(
        self,
        boxes: list[BoundingBox],
        img_width: float = 1000,
    ) -> list[Column]:
        """
        Detect columns by clustering x-coordinates.
        
        Parameters:
        -----------
        boxes : List of bounding boxes
        img_width : Image width for normalization
        
        Returns:
        --------
        List of Column objects sorted left to right
        """
        if len(boxes) < self.min_boxes_per_column:
            # Single column fallback
            return [Column(
                index=0,
                x_min=0,
                x_max=img_width,
                box_indices=list(range(len(boxes)))
            )]
        
        # Extract x-centers
        x_centers = np.array([[b.center[0]] for b in boxes])
        
        # Adaptive eps based on image width
        eps = max(self.min_column_gap, img_width * 0.05)
        
        # Cluster x-coordinates
        clustering = DBSCAN(eps=eps, min_samples=self.min_boxes_per_column)
        labels = clustering.fit_predict(x_centers)
        
        # Build columns from clusters
        columns = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
                
            indices = np.where(labels == label)[0].tolist()
            if len(indices) < self.min_boxes_per_column:
                continue
            
            cluster_boxes = [boxes[i] for i in indices]
            x_min = min(b.x_min for b in cluster_boxes)
            x_max = max(b.x_max for b in cluster_boxes)
            
            columns.append(Column(
                index=len(columns),
                x_min=x_min,
                x_max=x_max,
                box_indices=indices,
            ))
        
        # Sort columns left to right
        columns.sort(key=lambda c: c.center)
        
        # Re-index after sorting
        for i, col in enumerate(columns):
            col.index = i
        
        # Handle case where clustering fails
        if not columns:
            return [Column(
                index=0,
                x_min=0,
                x_max=img_width,
                box_indices=list(range(len(boxes)))
            )]
        
        # Assign unclustered boxes to nearest column
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            box = boxes[idx]
            nearest_col = min(columns, key=lambda c: abs(c.center - box.center[0]))
            nearest_col.box_indices.append(idx)
            # Update column bounds
            nearest_col.x_min = min(nearest_col.x_min, box.x_min)
            nearest_col.x_max = max(nearest_col.x_max, box.x_max)
        
        return columns
    
    def detect_price_column(
        self,
        columns: list[Column],
        boxes: list[BoundingBox],
        texts: list[str],
    ) -> Optional[int]:
        """
        Identify the price column.
        
        Criteria:
        - Rightmost column position
        - High digit ratio in contained text
        - Narrow width (prices are typically short)
        
        Returns:
        --------
        Index of price column, or None if not detected
        """
        if not columns:
            return None
        
        best_score = 0.0
        best_col_idx = None
        
        max_x = max(c.x_max for c in columns)
        avg_width = np.mean([c.width for c in columns])
        
        for col in columns:
            # Score based on position (rightmost preferred)
            position_score = col.x_max / max_x
            
            # Score based on digit ratio
            col_texts = [texts[i] for i in col.box_indices if i < len(texts)]
            if col_texts:
                digit_chars = sum(sum(1 for c in t if c.isdigit()) for t in col_texts)
                total_chars = sum(len(t) for t in col_texts)
                digit_ratio = digit_chars / max(total_chars, 1)
            else:
                digit_ratio = 0.0
            
            # Score based on width (narrow preferred for prices)
            width_score = 1.0 - min(col.width / avg_width, 1.0) if avg_width > 0 else 0.5
            
            # Combined score
            score = (
                0.4 * position_score +
                0.4 * digit_ratio +
                0.2 * width_score
            )
            
            if score > best_score and digit_ratio > 0.3:
                best_score = score
                best_col_idx = col.index
        
        # Mark the price column
        if best_col_idx is not None:
            columns[best_col_idx].is_price_column = True
        
        return best_col_idx
    
    def assign_box_to_column(
        self,
        box: BoundingBox,
        columns: list[Column],
    ) -> int:
        """Assign a box to the nearest column."""
        if not columns:
            return 0
        
        box_center = box.center[0]
        
        # Check if box is within any column
        for col in columns:
            if col.x_min <= box_center <= col.x_max:
                return col.index
        
        # Find nearest column
        return min(columns, key=lambda c: abs(c.center - box_center)).index
    
    def is_spanning_header(
        self,
        box: BoundingBox,
        columns: list[Column],
        span_threshold: float = 0.6,
    ) -> bool:
        """Check if a box spans multiple columns (likely a header)."""
        if len(columns) <= 1:
            return False
        
        total_width = columns[-1].x_max - columns[0].x_min
        box_width = box.width
        
        return box_width / total_width > span_threshold
