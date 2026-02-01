"""
Schema-compliant data models for menu extraction.
Designed to be domain-agnostic and traceable to source image.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class BoundingBox(BaseModel):
    """Bounding box coordinates from OCR."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height


class TextElementType(str, Enum):
    """Classification labels for text elements."""
    SECTION_HEADER = "section_header"      # Top-level category (e.g., "Wines", "Cocktails")
    GROUP_HEADER = "group_header"          # Sub-category (e.g., "Imported", "Domestic")
    ITEM_NAME = "item_name"                # Menu item name
    ITEM_PRICE = "item_price"              # Price value
    ITEM_DESCRIPTION = "item_description"  # Item description
    METADATA = "metadata"                  # Serving size, notes, etc.
    OTHER = "other"                        # Unclassified


class OCRResult(BaseModel):
    """Raw OCR extraction result with bounding box."""
    text: str
    bbox: BoundingBox
    confidence: float = 1.0
    
    class Config:
        frozen = True


class ClassifiedText(BaseModel):
    """OCR result with classification label."""
    ocr: OCRResult
    label: TextElementType
    label_confidence: float = 1.0
    
    @property
    def text(self) -> str:
        return self.ocr.text
    
    @property
    def bbox(self) -> BoundingBox:
        return self.ocr.bbox


class MenuItem(BaseModel):
    """A single menu item with traceable source."""
    name: str
    price: Optional[float] = None
    description: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
    
    # Traceability
    source_bboxes: Optional[list[BoundingBox]] = Field(default=None, exclude=True)


class MenuGroup(BaseModel):
    """A group of items under a sub-category."""
    id: str
    label: Optional[str] = None
    items: list[MenuItem] = Field(default_factory=list)
    meta: Optional[dict[str, Any]] = None
    
    # Traceability
    source_bbox: Optional[BoundingBox] = Field(default=None, exclude=True)


class MenuSection(BaseModel):
    """A top-level menu section/category."""
    id: str
    label: str
    groups: list[MenuGroup] = Field(default_factory=list)
    meta: Optional[dict[str, Any]] = None
    
    # Traceability
    source_bbox: Optional[BoundingBox] = Field(default=None, exclude=True)


class MenuDocument(BaseModel):
    """Complete extracted menu document."""
    menu: list[MenuSection] = Field(default_factory=list)
    
    # Metadata
    source_image: Optional[str] = Field(default=None, exclude=True)
    extraction_confidence: Optional[float] = Field(default=None, exclude=True)
    
    def to_output_json(self) -> dict:
        """Export to schema-compliant JSON (excludes traceability fields)."""
        return self.model_dump(exclude_none=True)


class ExtractionTrace(BaseModel):
    """Full extraction trace for debugging and validation."""
    source_image: str
    ocr_results: list[OCRResult]
    classified_texts: list[ClassifiedText]
    document: MenuDocument
    processing_time_ms: float
    warnings: list[str] = Field(default_factory=list)
