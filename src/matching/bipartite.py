"""
Global bipartite matching for price-item association using Hungarian algorithm.
Replaces greedy matching with optimal global assignment.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Optional
from dataclasses import dataclass

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
    """
    
    # Cost for infeasible matches
    INF_COST = 1e9
    
    def __init__(
        self,
        vertical_weight: float = 1.0,
        horizontal_penalty: float = 2.0,
        column_bonus: float = 0.5,
        max_vertical_distance: float = 2.0,
    ):
        """
        Initialize matcher.
        
        Parameters:
        -----------
        vertical_weight : Weight for vertical distance in cost
        horizontal_penalty : Penalty for price being left of item
        column_bonus : Bonus (negative cost) for price in price column
        max_vertical_distance : Maximum vertical distance (in line heights) for feasible match
        """
        self.vertical_weight = vertical_weight
        self.horizontal_penalty = horizontal_penalty
        self.column_bonus = column_bonus
        self.max_vertical_distance = max_vertical_distance
    
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
        
        Parameters:
        -----------
        items : List of classified text elements labeled as ITEM_NAME
        prices : List of classified text elements labeled as ITEM_PRICE
        columns : Optional column information
        img_width : Image width for normalization
        
        Returns:
        --------
        Cost matrix of shape (n_items, n_prices)
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
            item_x_center = (item_bbox.x_min + item_bbox.x_max) / 2
            
            for j, price in enumerate(prices):
                price_bbox = price.bbox
                price_y_center = (price_bbox.y_min + price_bbox.y_max) / 2
                price_x_center = (price_bbox.x_min + price_bbox.x_max) / 2
                
                # Check feasibility
                if not self._is_feasible(item_bbox, price_bbox, avg_height, columns, price_col_idx):
                    continue
                
                # Calculate cost components
                
                # 1. Vertical distance (normalized by average height)
                y_diff = abs(item_y_center - price_y_center) / avg_height
                vertical_cost = self.vertical_weight * y_diff
                
                # 2. Horizontal penalty (price should be right of item)
                if price_bbox.x_min < item_bbox.x_max:
                    # Price is left of or overlapping item - penalty
                    x_overlap = item_bbox.x_max - price_bbox.x_min
                    horizontal_cost = self.horizontal_penalty * (x_overlap / avg_height)
                else:
                    horizontal_cost = 0.0
                
                # 3. Column bonus (if price is in designated price column)
                column_cost = 0.0
                if columns and price_col_idx is not None:
                    price_col = self._get_column_for_box(price_bbox, columns)
                    if price_col == price_col_idx:
                        column_cost = -self.column_bonus  # Bonus = negative cost
                
                # 4. Distance bonus for closer prices
                x_distance = abs(price_x_center - item_x_center) / img_width
                distance_cost = 0.5 * x_distance
                
                # Total cost
                cost[i, j] = vertical_cost + horizontal_cost + column_cost + distance_cost
        
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
        3. No intervening structural elements (simplified)
        """
        item_y_center = (item_bbox.y_min + item_bbox.y_max) / 2
        price_y_center = (price_bbox.y_min + price_bbox.y_max) / 2
        
        # Constraint 1: Vertical alignment
        y_diff = abs(item_y_center - price_y_center)
        max_y_diff = self.max_vertical_distance * max(item_bbox.height, price_bbox.height)
        if y_diff > max_y_diff:
            return False
        
        # Constraint 2: Price not far left of item
        # Allow some overlap but not complete left placement
        if price_bbox.x_max < item_bbox.x_min - avg_height:
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
        
        Parameters:
        -----------
        items : List of item ClassifiedText
        prices : List of price ClassifiedText
        columns : Optional column information
        img_width : Image width for cost calculation
        
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
        
        Returns:
        --------
        List of MatchResult with cost information
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
                # Extract price value
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
        import re
        
        # Remove currency symbols
        cleaned = re.sub(r'[₹$£€¥]', '', text)
        cleaned = re.sub(r'(?:Rs\.?|INR)\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip().replace(',', '')
        
        # Handle ranges (take first value)
        if '-' in cleaned:
            cleaned = cleaned.split('-')[0]
        
        # Remove trailing non-digits
        cleaned = re.sub(r'[^0-9.]+$', '', cleaned)
        
        try:
            price = float(cleaned)
            if 0 < price < 100000:
                return price
        except (ValueError, TypeError):
            pass
        
        return None
