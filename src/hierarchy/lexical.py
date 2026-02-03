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
    
    # Section-level keywords (top-level categories) - expanded
    SECTION_INDICATORS = {
        # Food categories
        'appetizers', 'appetiser', 'appetisers', 'starters', 'starter',
        'mains', 'main course', 'main courses', 'entrees', 'entrées', 'entree',
        'desserts', 'dessert', 'sweets', 'puddings',
        'sides', 'side dishes', 'accompaniments',
        'salads', 'salad', 'soups', 'soup',
        'breakfast', 'lunch', 'dinner', 'brunch', 'supper',
        'specials', 'specialties', 'specialty', 'chef', "chef's",
        'seafood', 'fish', 'meat', 'meats', 'poultry', 'chicken',
        'vegetarian', 'vegan', 'veggie', 'vegetables',
        'pasta', 'pastas', 'pizza', 'pizzas', 'flatbreads',
        'sushi', 'sashimi', 'maki', 'nigiri',
        'curry', 'curries', 'tandoor', 'tandoori',
        'biryani', 'biryanis', 'rice', 'rices', 'pulao',
        'noodles', 'noodle', 'chow', 'wok',
        'grill', 'grills', 'grilled', 'bbq', 'barbeque',
        'burgers', 'burger', 'sandwiches', 'sandwich', 'subs',
        'wraps', 'wrap', 'rolls', 'roll',
        'snacks', 'snack', 'finger food', 'finger foods', 'nibbles',
        'mezze', 'tapas', 'small plates', 'sharing',
        'platters', 'platter', 'combos', 'combo', 'meals',
        
        # Beverage categories
        'beverages', 'beverage', 'drinks', 'drink',
        'cocktails', 'cocktail', 'mocktails', 'mocktail',
        'wines', 'wine', 'wine list', 'wine menu',
        'beers', 'beer', 'draft', 'draught', 'on tap',
        'spirits', 'spirit', 'liquors', 'liquor',
        'whiskey', 'whisky', 'whiskies', 'vodka', 'vodkas',
        'rum', 'rums', 'gin', 'gins', 'tequila',
        'brandy', 'cognac', 'liqueurs', 'liqueur',
        'coffee', 'coffees', 'espresso', 'tea', 'teas', 'chai',
        'juices', 'juice', 'fresh juice', 'smoothies', 'smoothie',
        'shakes', 'shake', 'milkshake', 'milkshakes',
        'soft drinks', 'sodas', 'soda', 'refreshments',
        'lemonade', 'lemonades', 'refreshers', 'coolers',
        'hot drinks', 'cold drinks', 'non-alcoholic', 'alcohol-free',
        'by the glass', 'by the bottle',
        
        # Other sections
        'menu', 'carte', 'à la carte', 'a la carte',
        'today', "today's", 'daily', 'seasonal',
        'featured', 'recommended', 'popular', 'favorites',
        'kids', 'children', "children's", "kids'", 'junior',
        'extras', 'add-ons', 'additions', 'supplements',
    }
    
    # Group-level keywords (sub-categories) - expanded
    GROUP_INDICATORS = {
        # Quality/type modifiers
        'domestic', 'imported', 'premium', 'deluxe', 'classic',
        'house', 'signature', 'special', 'featured', 'recommended',
        'single', 'double', 'triple', 'regular', 'large', 'small', 'medium',
        'half', 'full', 'quarter', 'whole', 'portion',
        
        # Dietary indicators
        'vegetarian', 'vegan', 'veggie',
        'gluten-free', 'gluten free', 'gf',
        'dairy-free', 'dairy free', 'lactose-free',
        'sugar-free', 'sugar free', 'low-calorie', 'diet',
        'organic', 'fresh', 'homemade', 'house-made', 'handcrafted',
        'healthy', 'light', 'lean',
        
        # Preparation styles
        'grilled', 'fried', 'deep fried', 'steamed', 'baked', 'roasted',
        'raw', 'smoked', 'cured', 'pickled', 'marinated',
        'hot', 'cold', 'iced', 'frozen', 'chilled', 'warm',
        'crispy', 'creamy', 'spicy', 'mild', 'medium',
        
        # Target audience
        'kids', 'children', 'family', 'sharing', 'for two',
        
        # Regional/cuisine types
        'indian', 'chinese', 'italian', 'mexican', 'thai', 'japanese',
        'american', 'continental', 'mediterranean', 'asian', 'european',
        'korean', 'vietnamese', 'french', 'spanish', 'greek',
        
        # Drink specifics
        'malts', 'malt', 'single malt', 'blended',
        'bourbon', 'scotch', 'irish', 'tennessee', 'rye',
        'red', 'white', 'rosé', 'rose', 'sparkling', 'champagne', 'prosecco',
        'draft', 'draught', 'bottled', 'canned', 'tap',
        'pint', 'glass', 'bottle', 'pitcher', 'carafe', 'jug',
        'shot', 'shots', 'neat', 'on the rocks',
        
        # Meal components
        'toppings', 'extras', 'add-ons', 'sides',
        'sauces', 'dressings', 'dips',
    }
    
    # Price-related patterns - expanded
    PRICE_PATTERNS = [
        r'^[\$£€₹¥]?\s*\d+(?:[.,]\d{1,2})?\s*$',  # $12.99, ₹500
        r'^\d{2,5}$',  # 500, 1200 (common in Asian menus)
        r'^[\$£€₹¥]\s*\d+',  # $12, ₹500
        r'^\d+\s*[\$£€₹¥]$',  # 12$, 500₹
        r'^(?:Rs\.?|INR|USD|EUR|GBP)\s*\d+',  # Rs. 500
        r'^\d+\s*/-',  # 500/- (Indian style)
        r'^\d+/\d+',  # 500/700 (price range)
        r'^[\d,]+(?:\.\d{2})?$',  # 1,200.00
    ]
    
    # Description indicators - expanded
    DESCRIPTION_PATTERNS = [
        r'served with',
        r'comes with',
        r'includes',
        r'topped with',
        r'made with',
        r'prepared with',
        r'cooked in',
        r'fresh\s+\w+',
        r'homemade',
        r'our\s+\w+',
        r'a\s+\w+\s+of',
        r'blend\s+of',
        r'choice of',
        r'selection of',
        r'assortment of',
        r'combination of',
        r'marinated in',
        r'seasoned with',
        r'garnished with',
        r'drizzled with',
        r'stuffed with',
        r'layered with',
        r'accompanied by',
        r'paired with',
    ]
    
    def __init__(
        self,
        section_boost: float = 0.35,
        group_boost: float = 0.30,
        price_boost: float = 0.45,
        description_boost: float = 0.25,
    ):
        """
        Initialize lexical priors with tuned boost values.
        """
        self.section_boost = section_boost
        self.group_boost = group_boost
        self.price_boost = price_boost
        self.description_boost = description_boost
        
        # Compile patterns for speed
        self.price_patterns = [re.compile(p, re.IGNORECASE) for p in self.PRICE_PATTERNS]
        self.description_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DESCRIPTION_PATTERNS
        ]
    
    def compute_prior(
        self,
        text: str,
    ) -> dict[TextElementType, float]:
        """
        Compute prior probability adjustments for each label.
        """
        priors = {t: 0.0 for t in TextElementType}
        
        text_lower = text.lower().strip()
        # Use word boundary matching for better accuracy
        words = set(re.findall(r'\b\w+\b', text_lower))
        
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
                0.25
            )
        
        # Very short text with high digit ratio -> likely price
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        if len(text) <= 10 and digit_ratio > 0.4:
            priors[TextElementType.ITEM_PRICE] = max(
                priors[TextElementType.ITEM_PRICE],
                0.35
            )
        
        # Long text -> likely description or item name
        if len(words) > 6:
            priors[TextElementType.ITEM_DESCRIPTION] = max(
                priors[TextElementType.ITEM_DESCRIPTION],
                0.20
            )
        
        # Text ending with ellipsis or continuation -> description
        if text.endswith('...') or text.endswith('…'):
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
        """Get most likely label if prior is strong enough."""
        priors = self.compute_prior(text)
        
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
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return bool(words & self.SECTION_INDICATORS)
    
    def is_group_keyword(self, text: str) -> bool:
        """Check if text contains group keywords."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return bool(words & self.GROUP_INDICATORS)
