"""
Font scale analysis for hierarchy detection.
Clusters text elements by font size to identify hierarchy levels.
"""

import numpy as np
from typing import Optional
from sklearn.cluster import KMeans
from dataclasses import dataclass

from ..models.schema import BoundingBox, OCRResult


@dataclass
class FontLevel:
    """Represents a detected font size level."""
    level: int  # 0 = largest (headers), higher = smaller
    mean_height: float
    box_indices: list[int]


class FontScaleAnalyzer:
    """
    Detect hierarchy via relative font scaling.
    
    Menu documents typically use 3-4 font size levels:
    - Level 0: Section headers (largest)
    - Level 1: Group headers  
    - Level 2: Item names
    - Level 3: Descriptions/metadata (smallest)
    """
    
    def __init__(
        self,
        n_levels: int = 4,
        min_ratio_for_header: float = 1.2,
    ):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        n_levels : Maximum number of font levels to detect
        min_ratio_for_header : Minimum height ratio vs mean to be considered header
        """
        self.n_levels = n_levels
        self.min_ratio_for_header = min_ratio_for_header
    
    def analyze(self, boxes: list[BoundingBox]) -> list[int]:
        """
        Assign font level to each box based on height clustering.
        
        Parameters:
        -----------
        boxes : List of bounding boxes
        
        Returns:
        --------
        List of level indices (0 = largest/header, higher = smaller)
        """
        if not boxes:
            return []
        
        if len(boxes) == 1:
            return [1]  # Default to item level
        
        # Extract heights
        heights = np.array([b.height for b in boxes]).reshape(-1, 1)
        
        # Determine optimal number of clusters
        n_clusters = min(self.n_levels, len(set(heights.flatten())))
        n_clusters = max(2, n_clusters)  # At least 2 levels
        
        # Handle case where all heights are similar
        height_std = np.std(heights)
        if height_std < 2.0:  # Very uniform heights
            return [2] * len(boxes)  # All item level
        
        # Cluster heights
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(heights)
        
        # Map clusters to levels (larger height = lower level number)
        cluster_means = {}
        for i in range(n_clusters):
            mask = labels == i
            if mask.any():
                cluster_means[i] = heights[mask].mean()
        
        # Sort clusters by mean height (descending)
        sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
        
        # Create mapping from cluster to level
        cluster_to_level = {}
        for level, (cluster_id, _) in enumerate(sorted_clusters):
            cluster_to_level[cluster_id] = level
        
        # Assign levels
        levels = [cluster_to_level[label] for label in labels]
        
        return levels
    
    def get_font_levels(self, boxes: list[BoundingBox]) -> list[FontLevel]:
        """
        Get detailed font level information.
        
        Returns:
        --------
        List of FontLevel objects with statistics
        """
        if not boxes:
            return []
        
        levels = self.analyze(boxes)
        heights = [b.height for b in boxes]
        
        # Group by level
        level_data = {}
        for i, (level, height) in enumerate(zip(levels, heights)):
            if level not in level_data:
                level_data[level] = {'heights': [], 'indices': []}
            level_data[level]['heights'].append(height)
            level_data[level]['indices'].append(i)
        
        # Create FontLevel objects
        result = []
        for level in sorted(level_data.keys()):
            data = level_data[level]
            result.append(FontLevel(
                level=level,
                mean_height=np.mean(data['heights']),
                box_indices=data['indices']
            ))
        
        return result
    
    def compute_scale_ratios(
        self,
        boxes: list[BoundingBox],
        window_size: int = 3,
    ) -> list[float]:
        """
        Compute height ratio vs local neighbors for each box.
        
        Useful for detecting headers that are larger than surrounding text.
        
        Parameters:
        -----------
        boxes : List of bounding boxes
        window_size : Number of neighbors to consider
        
        Returns:
        --------
        List of ratios (>1 means larger than neighbors)
        """
        if not boxes:
            return []
        
        heights = [b.height for b in boxes]
        n = len(heights)
        
        if n == 1:
            return [1.0]
        
        ratios = []
        for i in range(n):
            # Get neighbor heights
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            neighbor_heights = heights[start:i] + heights[i+1:end]
            
            if neighbor_heights:
                mean_neighbor = np.mean(neighbor_heights)
                ratio = heights[i] / mean_neighbor if mean_neighbor > 0 else 1.0
            else:
                ratio = 1.0
            
            ratios.append(ratio)
        
        return ratios
    
    def is_likely_header(
        self,
        box: BoundingBox,
        all_boxes: list[BoundingBox],
        idx: int,
    ) -> tuple[bool, float]:
        """
        Determine if a box is likely a header based on font size.
        
        Returns:
        --------
        Tuple of (is_header, confidence)
        """
        if not all_boxes:
            return False, 0.0
        
        heights = [b.height for b in all_boxes]
        mean_height = np.mean(heights)
        std_height = np.std(heights)
        
        box_height = box.height
        
        # Z-score approach
        if std_height > 0:
            z_score = (box_height - mean_height) / std_height
            is_header = z_score > 1.0  # 1 std above mean
            confidence = min(1.0, max(0.0, z_score / 2.0))  # Normalize
        else:
            # All same height
            is_header = False
            confidence = 0.0
        
        # Also check ratio
        ratio = box_height / mean_height if mean_height > 0 else 1.0
        if ratio > self.min_ratio_for_header:
            is_header = True
            confidence = max(confidence, min(1.0, (ratio - 1.0) / 0.5))
        
        return is_header, confidence
