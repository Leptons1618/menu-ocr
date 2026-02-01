"""
Reading order resolution for multi-column documents.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..models.schema import BoundingBox
from .columns import Column


@dataclass
class ReadingOrderItem:
    """Item with reading order metadata."""
    original_index: int
    column_index: int
    y_position: float
    is_spanning: bool = False


class ReadingOrderResolver:
    """Establish reading order respecting column structure."""
    
    def __init__(self, line_tolerance: float = 0.5):
        """
        Initialize resolver.
        
        Parameters:
        -----------
        line_tolerance : Tolerance for considering boxes on same line (as fraction of height)
        """
        self.line_tolerance = line_tolerance
    
    def resolve(
        self,
        boxes: list[BoundingBox],
        columns: list[Column],
        texts: Optional[list[str]] = None,
    ) -> list[int]:
        """
        Determine reading order for boxes respecting columns.
        
        Strategy:
        1. Identify spanning headers (process first)
        2. For each row of content, process columns left-to-right
        3. Within each column, process top-to-bottom
        
        Parameters:
        -----------
        boxes : List of bounding boxes
        columns : Detected columns
        texts : Optional text content for header detection
        
        Returns:
        --------
        List of indices in reading order
        """
        if not boxes:
            return []
        
        if len(columns) <= 1:
            # Single column: simple top-to-bottom order
            return sorted(range(len(boxes)), key=lambda i: (boxes[i].y_min, boxes[i].x_min))
        
        # Categorize boxes
        spanning = []
        by_column = {col.index: [] for col in columns}
        
        for i, box in enumerate(boxes):
            if self._is_spanning(box, columns):
                spanning.append(ReadingOrderItem(
                    original_index=i,
                    column_index=-1,
                    y_position=box.y_min,
                    is_spanning=True
                ))
            else:
                col_idx = self._assign_to_column(box, columns)
                by_column[col_idx].append(ReadingOrderItem(
                    original_index=i,
                    column_index=col_idx,
                    y_position=box.y_min,
                    is_spanning=False
                ))
        
        # Sort items within each column by y-position
        for col_idx in by_column:
            by_column[col_idx].sort(key=lambda item: item.y_position)
        
        # Sort spanning headers by y-position
        spanning.sort(key=lambda item: item.y_position)
        
        # Build reading order
        result = []
        
        # Process rows from top to bottom
        all_items = spanning + [item for items in by_column.values() for item in items]
        all_items.sort(key=lambda item: item.y_position)
        
        # Group into rows
        rows = self._group_into_rows(all_items, boxes)
        
        for row in rows:
            # Spanning items first
            spanning_in_row = [item for item in row if item.is_spanning]
            spanning_in_row.sort(key=lambda item: item.y_position)
            result.extend([item.original_index for item in spanning_in_row])
            
            # Then non-spanning items by column order (left to right)
            non_spanning = [item for item in row if not item.is_spanning]
            non_spanning.sort(key=lambda item: (item.column_index, item.y_position))
            result.extend([item.original_index for item in non_spanning])
        
        return result
    
    def _is_spanning(self, box: BoundingBox, columns: list[Column]) -> bool:
        """Check if box spans multiple columns."""
        if len(columns) <= 1:
            return False
        
        # Count how many columns the box overlaps
        overlapping = 0
        for col in columns:
            # Check horizontal overlap
            if box.x_min < col.x_max and box.x_max > col.x_min:
                overlapping += 1
        
        return overlapping > 1
    
    def _assign_to_column(self, box: BoundingBox, columns: list[Column]) -> int:
        """Assign box to best matching column."""
        box_center = box.center[0]
        
        # Find column with most overlap or nearest center
        best_col = 0
        best_overlap = 0
        
        for col in columns:
            # Calculate overlap
            overlap_start = max(box.x_min, col.x_min)
            overlap_end = min(box.x_max, col.x_max)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = col.index
        
        # If no overlap, use nearest center
        if best_overlap == 0:
            best_col = min(columns, key=lambda c: abs(c.center - box_center)).index
        
        return best_col
    
    def _group_into_rows(
        self,
        items: list[ReadingOrderItem],
        boxes: list[BoundingBox],
    ) -> list[list[ReadingOrderItem]]:
        """Group items into visual rows."""
        if not items:
            return []
        
        # Sort by y-position
        sorted_items = sorted(items, key=lambda item: item.y_position)
        
        rows = []
        current_row = [sorted_items[0]]
        current_y = sorted_items[0].y_position
        
        for item in sorted_items[1:]:
            box = boxes[item.original_index]
            height = box.height
            tolerance = height * self.line_tolerance
            
            if abs(item.y_position - current_y) <= tolerance:
                # Same row
                current_row.append(item)
            else:
                # New row
                rows.append(current_row)
                current_row = [item]
                current_y = item.y_position
        
        if current_row:
            rows.append(current_row)
        
        return rows
