"""
Lightweight lexical priors for menu element classification.
Domain-specific keywords that signal section/group headers.
"""

import re
from typing import Optional
from ..models.schema import TextElementType


class LexicalPriors:
    """
    Domain-specific lexical signals for menu classification.
    
    Provides prior probability adjustments based on keyword matching.
    """
    
    # Section-level keywords (top-level categories)
    SECTION_INDICATORS = {
        # Food categories
        'appetizers', 'starters', 'mains', 'entrees', 'entrées',
        'desserts', 'dessert', 'sides', 'salads', 'soups',
        'breakfast', 'lunch', 'dinner', 'brunch',
        'specials', 'specialties', 'chef',
        'seafood', 'meat', 'poultry', 'vegetarian', 'vegan',
        'pasta', 'pizza', 'sushi', 'curry', 'tandoor',
        'grill', 'grilled', 'fried', 'roasted',
        
        # Beverage categories
        'beverages', 'drinks', 'cocktails', 'mocktails',
        'wines', 'wine', 'beer', 'beers', 'spirits',
        'whiskey', 'whisky', 'vodka', 'rum', 'gin', 'tequila',
        'coffee', 'tea', 'juices', 'smoothies',
        'soft drinks', 'sodas',
        
        # Other sections
        'menu', 'carte', 'today',
    }
    
    # Group-level keywords (sub-categories)
    GROUP_INDICATORS = {
        # Quality/type modifiers
        'domestic', 'imported', 'premium', 'deluxe', 'classic',
        'house', 'signature', 'special', 'featured',
        'single', 'double', 'large', 'small', 'regular',
        
        # Dietary indicators
        'vegetarian', 'vegan', 'gluten-free', 'gluten free',
        'dairy-free', 'sugar-free', 'low-calorie',
        'organic', 'fresh', 'homemade',
        
        # Preparation styles
        'grilled', 'fried', 'steamed', 'baked', 'raw',
        'hot', 'cold', 'iced', 'frozen',
        
        # Target audience
        'kids', 'children', 'family',
        
        # Regional
        'indian', 'chinese', 'italian', 'mexican', 'thai',
        'japanese', 'american', 'continental',
        
        # Drink specifics
        'malts', 'malt', 'bourbon', 'scotch', 'tennessee',
        'red', 'white', 'rosé', 'sparkling', 'champagne',
        'draft', 'draught', 'bottled', 'canned',
        'pint', 'glass', 'bottle', 'pitcher',
    }
    
    # Price-related patterns
    PRICE_PATTERNS = [
        r'^[\$£€₹¥]?\s*\d+(?:[.,]\d{1,2})?\s*$',  # $12.99, ₹500
        r'^\d{2,5}$',  # 500, 1200 (common in Asian menus)
        r'^[\$£€₹¥]\s*\d+',  # $12, ₹500
        r'^\d+\s*[\$£€₹¥]$',  # 12$, 500₹
        r'^(?:Rs\.?|INR|USD|EUR)\s*\d+',  # Rs. 500
    ]
    
    # Description indicators
    DESCRIPTION_PATTERNS = [
        r'served with',
        r'comes with',
        r'includes',
        r'topped with',
        r'made with',
        r'fresh\s+\w+',
        r'homemade',
        r'our\s+\w+',
        r'a\s+\w+\s+of',
        r'blend\s+of',
    ]
    
    def __init__(
        self,
        section_boost: float = 0.3,
        group_boost: float = 0.25,
        price_boost: float = 0.4,
        description_boost: float = 0.2,
    ):
        """
        Initialize lexical priors.
        
        Parameters:
        -----------
        section_boost : Prior boost for section header detection
        group_boost : Prior boost for group header detection
        price_boost : Prior boost for price detection
        description_boost : Prior boost for description detection
        """
        self.section_boost = section_boost
        self.group_boost = group_boost
        self.price_boost = price_boost
        self.description_boost = description_boost
        
        # Compile patterns
        self.price_patterns = [re.compile(p) for p in self.PRICE_PATTERNS]
        self.description_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DESCRIPTION_PATTERNS
        ]
    
    def compute_prior(
        self,
        text: str,
    ) -> dict[TextElementType, float]:
        """
        Compute prior probability adjustments for each label.
        
        Parameters:
        -----------
        text : Text content to analyze
        
        Returns:
        --------
        Dict mapping TextElementType to prior adjustment (0.0 = no change)
        """
        priors = {t: 0.0 for t in TextElementType}
        
        text_lower = text.lower().strip()
        words = set(text_lower.split())
        
        # Check for section indicators
        if words & self.SECTION_INDICATORS:
            priors[TextElementType.SECTION_HEADER] = self.section_boost
            # Slight boost to group header too (often overlapping vocabulary)
            priors[TextElementType.GROUP_HEADER] = self.section_boost * 0.3
        
        # Check for group indicators
        if words & self.GROUP_INDICATORS:
            priors[TextElementType.GROUP_HEADER] = max(
                priors[TextElementType.GROUP_HEADER],
                self.group_boost
            )
        
        # Check for price patterns
        if any(p.match(text.strip()) for p in self.price_patterns):
            priors[TextElementType.ITEM_PRICE] = self.price_boost
        
        # Check for description patterns
        if any(p.search(text_lower) for p in self.description_patterns):
            priors[TextElementType.ITEM_DESCRIPTION] = self.description_boost
        
        # Heuristics based on text structure
        
        # All caps short text -> likely header
        if text.isupper() and len(words) <= 4 and len(text) > 3:
            priors[TextElementType.SECTION_HEADER] = max(
                priors[TextElementType.SECTION_HEADER],
                0.2
            )
        
        # Very short text with digits -> likely price
        if len(text) <= 8 and sum(c.isdigit() for c in text) / max(len(text), 1) > 0.5:
            priors[TextElementType.ITEM_PRICE] = max(
                priors[TextElementType.ITEM_PRICE],
                0.3
            )
        
        # Long text -> likely description or item name
        if len(words) > 6:
            priors[TextElementType.ITEM_DESCRIPTION] = max(
                priors[TextElementType.ITEM_DESCRIPTION],
                0.15
            )
        
        return priors
    
    def get_likely_label(
        self,
        text: str,
        threshold: float = 0.2,
    ) -> Optional[TextElementType]:
        """
        Get most likely label if prior is strong enough.
        
        Parameters:
        -----------
        text : Text to analyze
        threshold : Minimum prior to return a label
        
        Returns:
        --------
        Most likely TextElementType or None if no strong signal
        """
        priors = self.compute_prior(text)
        
        # Find max prior
        best_label = max(priors, key=priors.get)
        best_prior = priors[best_label]
        
        if best_prior >= threshold:
            return best_label
        
        return None
    
    def is_price_text(self, text: str) -> bool:
        """Check if text matches price patterns."""
        return any(p.match(text.strip()) for p in self.price_patterns)
    
    def is_section_keyword(self, text: str) -> bool:
        """Check if text contains section keywords."""
        words = set(text.lower().split())
        return bool(words & self.SECTION_INDICATORS)
    
    def is_group_keyword(self, text: str) -> bool:
        """Check if text contains group keywords."""
        words = set(text.lower().split())
        return bool(words & self.GROUP_INDICATORS)
