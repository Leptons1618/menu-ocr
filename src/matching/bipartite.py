"""
Global bipartite matching for price-item association using Hungarian algorithm.
Replaces greedy matching with optimal global assignment.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional
from dataclasses import dataclass
import re

from ..models.schema import ClassifiedText, TextElementType, BoundingBox
from ..layout.columns import Column


@dataclass
class MatchResult:
    """Result of price-item matching."""
    item_index: int
    price_index: int
    cost: float
    price_value: Optional[float] = None


class PriceItemMatcher:
    """
    Global optimal matching using Hungarian algorithm.
    
    Matches items to prices considering:
    - Vertical alignment (items and prices on same line)
    - Horizontal positioning (prices typically right of items)
    - Column consistency (prices in price column or same column as item)
    - Line-based proximity (prefer prices on same visual line)
    """
    
    # Cost for infeasible matches
    INF_COST = 1e9
    
    # Price extraction patterns
    PRICE_PATTERNS = [
        r'[\$£€₹¥]?\s*(\d+(?:[.,]\d{1,2})?)',
        r'(\d+(?:[.,]\d{2}))',
        r'(?:Rs\.?|INR)\s*(\d+)',
        r'(\d+)\s*/-',
    ]
    
    def __init__(
        self,
        vertical_weight: float = 1.5,
        horizontal_penalty: float = 3.0,
        column_bonus: float = 0.8,
        max_vertical_distance: float = 1.5,
        same_line_bonus: float = 1.0,
    ):
        """
        Initialize matcher with optimized weights.
        
        Parameters:
        -----------
        vertical_weight : Weight for vertical distance in cost
        horizontal_penalty : Penalty for price being left of item
        column_bonus : Bonus (negative cost) for price in price column
        max_vertical_distance : Maximum vertical distance (in line heights) for feasible match
        same_line_bonus : Bonus for items and prices on the same visual line
        """
        self.vertical_weight = vertical_weight
        self.horizontal_penalty = horizontal_penalty
        self.column_bonus = column_bonus
        self.max_vertical_distance = max_vertical_distance
        self.same_line_bonus = same_line_bonus
        
        # Compile price patterns
        self._price_patterns = [re.compile(p, re.IGNORECASE) for p in self.PRICE_PATTERNS]
    
    def build_cost_matrix(
        self,
        items: list[ClassifiedText],
        prices: list[ClassifiedText],
        columns: Optional[list[Column]] = None,
        img_width: float = 1000,
    ) -> np.ndarray:
        """
        Build cost matrix for Hungarian algorithm.
        
        cost[i,j] represents cost of assigning item i to price j.
        Lower cost = better match.
        """
        n_items = len(items)
        n_prices = len(prices)
        
        if n_items == 0 or n_prices == 0:
            return np.array([]).reshape(n_items, n_prices)
        
        # Initialize with infinite cost
        cost = np.full((n_items, n_prices), self.INF_COST)
        
        # Calculate average height for normalization
        all_heights = [item.bbox.height for item in items] + [price.bbox.height for price in prices]
        avg_height = np.mean(all_heights) if all_heights else 20.0
        
        # Find price column index
        price_col_idx = None
        if columns:
            for col in columns:
                if col.is_price_column:
                    price_col_idx = col.index
                    break
        
        for i, item in enumerate(items):
            item_bbox = item.bbox
            item_y_center = (item_bbox.y_min + item_bbox.y_max) / 2
            item_x_right = item_bbox.x_max
            
            for j, price in enumerate(prices):
                price_bbox = price.bbox
                price_y_center = (price_bbox.y_min + price_bbox.y_max) / 2
                price_x_left = price_bbox.x_min
                
                # Check feasibility
                if not self._is_feasible(item_bbox, price_bbox, avg_height, columns, price_col_idx):
                    continue
                
                # Calculate cost components
                
                # 1. Vertical distance (normalized by average height)
                y_diff = abs(item_y_center - price_y_center) / avg_height
                vertical_cost = self.vertical_weight * y_diff
                
                # 2. Same line bonus (strong preference for vertically aligned pairs)
                max_height = max(item_bbox.height, price_bbox.height)
                if abs(item_y_center - price_y_center) < max_height * 0.5:
                    # Items are on the same visual line
                    same_line_bonus = -self.same_line_bonus
                else:
                    same_line_bonus = 0.0
                
                # 3. Horizontal penalty (price should be right of item)
                if price_x_left < item_x_right:
                    # Price is left of or overlapping item - penalty
                    x_overlap = item_x_right - price_x_left
                    horizontal_cost = self.horizontal_penalty * (x_overlap / avg_height)
                else:
                    # Price is to the right (good) - small cost based on distance
                    horizontal_cost = 0.1 * (price_x_left - item_x_right) / img_width
                
                # 4. Column bonus (if price is in designated price column)
                column_cost = 0.0
                if columns and price_col_idx is not None:
                    price_col = self._get_column_for_box(price_bbox, columns)
                    if price_col == price_col_idx:
                        column_cost = -self.column_bonus  # Bonus = negative cost
                
                # 5. Relative position bonus (price should be at similar y-level)
                y_overlap = max(0, min(item_bbox.y_max, price_bbox.y_max) - 
                               max(item_bbox.y_min, price_bbox.y_min))
                overlap_ratio = y_overlap / max(item_bbox.height, price_bbox.height, 1)
                overlap_bonus = -0.3 * overlap_ratio  # More overlap = lower cost
                
                # Total cost
                cost[i, j] = max(0.01, vertical_cost + horizontal_cost + column_cost + 
                                same_line_bonus + overlap_bonus)
        
        return cost
    
    def _is_feasible(
        self,
        item_bbox: BoundingBox,
        price_bbox: BoundingBox,
        avg_height: float,
        columns: Optional[list[Column]],
        price_col_idx: Optional[int],
    ) -> bool:
        """
        Check if item-price pair is a feasible match.
        
        Constraints:
        1. Price must be vertically aligned (within max_vertical_distance)
        2. Price should not be far to the left of item
        """
        item_y_center = (item_bbox.y_min + item_bbox.y_max) / 2
        price_y_center = (price_bbox.y_min + price_bbox.y_max) / 2
        
        # Constraint 1: Vertical alignment (relaxed for better matching)
        y_diff = abs(item_y_center - price_y_center)
        max_y_diff = self.max_vertical_distance * max(item_bbox.height, price_bbox.height)
        if y_diff > max_y_diff:
            return False
        
        # Constraint 2: Price not far left of item (some overlap allowed)
        if price_bbox.x_max < item_bbox.x_min - avg_height * 0.5:
            return False
        
        return True
    
    def _get_column_for_box(
        self,
        bbox: BoundingBox,
        columns: list[Column],
    ) -> Optional[int]:
        """Get column index for a bounding box."""
        if not columns:
            return None
        
        box_center = (bbox.x_min + bbox.x_max) / 2
        
        for col in columns:
            if col.x_min <= box_center <= col.x_max:
                return col.index
        
        # Find nearest
        return min(columns, key=lambda c: abs(c.center - box_center)).index
    
    def match(
        self,
        items: list[ClassifiedText],
        prices: list[ClassifiedText],
        columns: Optional[list[Column]] = None,
        img_width: float = 1000,
    ) -> dict[int, int]:
        """
        Find optimal item-price matching using Hungarian algorithm.
        
        Returns:
        --------
        Dictionary mapping item indices to price indices
        """
        if not items or not prices:
            return {}
        
        # Build cost matrix
        cost = self.build_cost_matrix(items, prices, columns, img_width)
        
        if cost.size == 0:
            return {}
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Filter out infeasible assignments
        matches = {}
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < self.INF_COST:
                matches[i] = j
        
        return matches
    
    def match_with_details(
        self,
        items: list[ClassifiedText],
        prices: list[ClassifiedText],
        columns: Optional[list[Column]] = None,
        img_width: float = 1000,
    ) -> list[MatchResult]:
        """
        Find optimal matching with detailed results.
        """
        if not items or not prices:
            return []
        
        cost = self.build_cost_matrix(items, prices, columns, img_width)
        
        if cost.size == 0:
            return []
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        results = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < self.INF_COST:
                price_value = self._extract_price_value(prices[j].text)
                results.append(MatchResult(
                    item_index=i,
                    price_index=j,
                    cost=cost[i, j],
                    price_value=price_value
                ))
        
        return results
    
    def _extract_price_value(self, text: str) -> Optional[float]:
        """Extract numeric price from text."""
        # Remove currency symbols and clean up
        cleaned = text.strip()
        cleaned = re.sub(r'[₹$£€¥]', '', cleaned)
        cleaned = re.sub(r'(?:Rs\.?|INR|USD|EUR|GBP)\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace(',', '').replace(' ', '')
        
        # Handle /- suffix (Indian style)
        cleaned = re.sub(r'/-\s*$', '', cleaned)
        
        # Handle ranges (take first value)
        if '-' in cleaned or '/' in cleaned:
            cleaned = re.split(r'[-/]', cleaned)[0]
        
        # Remove trailing non-digits
        cleaned = re.sub(r'[^0-9.]+$', '', cleaned)
        cleaned = re.sub(r'^[^0-9.]+', '', cleaned)
        
        try:
            price = float(cleaned)
            if 0 < price < 100000:
                return price
        except (ValueError, TypeError):
            pass
        
        return None
