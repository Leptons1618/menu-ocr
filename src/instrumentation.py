"""
Pipeline instrumentation for tracing, logging, and diagnostics.
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

from .models.schema import BoundingBox, OCRResult, ClassifiedText


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    stage_name: str
    start_time: float
    end_time: float
    input_count: int
    output_count: int
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class PipelineTrace:
    """Complete trace of pipeline execution for debugging and analysis."""
    
    # Identification
    image_path: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    trace_id: str = ""
    config_hash: str = ""
    random_seed: int = 42
    
    # Per-stage timing (ms)
    ocr_detection_ms: float = 0.0
    ocr_recognition_ms: float = 0.0
    column_detection_ms: float = 0.0
    classification_ms: float = 0.0
    hierarchy_decoding_ms: float = 0.0
    matching_ms: float = 0.0
    structure_building_ms: float = 0.0
    total_ms: float = 0.0
    
    # Per-stage counts
    detection_count: int = 0
    ocr_count: int = 0
    column_count: int = 0
    item_count: int = 0
    price_count: int = 0
    match_count: int = 0
    
    # Detailed outputs (optional, for debugging)
    detections: list[dict] = field(default_factory=list)
    ocr_results: list[dict] = field(default_factory=list)
    columns: list[dict] = field(default_factory=list)
    classifications: list[dict] = field(default_factory=list)
    matches: list[dict] = field(default_factory=list)
    
    # Per-stage metrics (if ground truth available)
    ocr_recall: Optional[float] = None
    classification_accuracy: Optional[float] = None
    matching_accuracy: Optional[float] = None
    
    # Warnings and errors
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = self._generate_trace_id()
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        content = f"{self.image_path}_{self.timestamp}_{self.random_seed}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(f"[{datetime.utcnow().isoformat()}] {message}")
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(f"[{datetime.utcnow().isoformat()}] {message}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, output_dir: Path):
        """Save trace to file with deterministic filename."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"trace_{self.trace_id}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        
        return filepath
    
    def get_timing_breakdown(self) -> dict[str, float]:
        """Get timing breakdown by stage."""
        return {
            'ocr_detection': self.ocr_detection_ms,
            'ocr_recognition': self.ocr_recognition_ms,
            'column_detection': self.column_detection_ms,
            'classification': self.classification_ms,
            'hierarchy_decoding': self.hierarchy_decoding_ms,
            'matching': self.matching_ms,
            'structure_building': self.structure_building_ms,
            'total': self.total_ms,
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Pipeline Trace: {self.trace_id}",
            f"Image: {self.image_path}",
            f"Total time: {self.total_ms:.1f}ms",
            f"",
            f"Stage Timing:",
            f"  OCR Detection:    {self.ocr_detection_ms:7.1f}ms ({self.detection_count} boxes)",
            f"  OCR Recognition:  {self.ocr_recognition_ms:7.1f}ms ({self.ocr_count} results)",
            f"  Column Detection: {self.column_detection_ms:7.1f}ms ({self.column_count} columns)",
            f"  Classification:   {self.classification_ms:7.1f}ms",
            f"  Hierarchy:        {self.hierarchy_decoding_ms:7.1f}ms",
            f"  Matching:         {self.matching_ms:7.1f}ms ({self.match_count} matches)",
            f"  Structure:        {self.structure_building_ms:7.1f}ms",
            f"",
            f"Counts:",
            f"  Items:  {self.item_count}",
            f"  Prices: {self.price_count}",
        ]
        
        if self.warnings:
            lines.append(f"\nWarnings: {len(self.warnings)}")
            for w in self.warnings[:5]:
                lines.append(f"  - {w}")
        
        if self.errors:
            lines.append(f"\nErrors: {len(self.errors)}")
            for e in self.errors[:5]:
                lines.append(f"  - {e}")
        
        return '\n'.join(lines)


class PipelineInstrumentor:
    """Helper class for instrumenting pipeline execution."""
    
    def __init__(self, image_path: str, config: Optional[dict] = None):
        """
        Initialize instrumentor.
        
        Parameters:
        -----------
        image_path : Path to input image
        config : Pipeline configuration dict
        """
        self.trace = PipelineTrace(image_path=str(image_path))
        self._stage_start = None
        self._total_start = time.time()
        
        if config:
            config_str = json.dumps(config, sort_keys=True, default=str)
            self.trace.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    
    def start_stage(self, stage_name: str):
        """Mark start of a pipeline stage."""
        self._stage_start = time.time()
        self._current_stage = stage_name
    
    def end_stage(self, stage_name: str, input_count: int = 0, output_count: int = 0):
        """Mark end of a pipeline stage and record timing."""
        if self._stage_start is None:
            return
        
        elapsed_ms = (time.time() - self._stage_start) * 1000
        
        # Store timing based on stage name
        stage_attr = f"{stage_name}_ms"
        if hasattr(self.trace, stage_attr):
            setattr(self.trace, stage_attr, elapsed_ms)
        
        self._stage_start = None
    
    def record_detections(self, boxes: list[BoundingBox]):
        """Record detection results."""
        self.trace.detection_count = len(boxes)
        self.trace.detections = [
            {
                'x_min': b.x_min, 'y_min': b.y_min,
                'x_max': b.x_max, 'y_max': b.y_max
            }
            for b in boxes
        ]
    
    def record_ocr_results(self, results: list[OCRResult]):
        """Record OCR results."""
        self.trace.ocr_count = len(results)
        self.trace.ocr_results = [
            {
                'text': r.text,
                'confidence': r.confidence,
                'bbox': {
                    'x_min': r.bbox.x_min, 'y_min': r.bbox.y_min,
                    'x_max': r.bbox.x_max, 'y_max': r.bbox.y_max
                }
            }
            for r in results
        ]
    
    def record_columns(self, columns: list):
        """Record column detection results."""
        self.trace.column_count = len(columns)
        self.trace.columns = [
            {
                'index': c.index,
                'x_min': c.x_min,
                'x_max': c.x_max,
                'is_price_column': c.is_price_column,
                'box_count': len(c.box_indices)
            }
            for c in columns
        ]
    
    def record_classifications(self, classified: list[ClassifiedText]):
        """Record classification results."""
        self.trace.classifications = [
            {
                'text': c.text,
                'label': c.label.value,
                'confidence': c.label_confidence
            }
            for c in classified
        ]
        
        # Count items and prices
        self.trace.item_count = sum(
            1 for c in classified if c.label.value == 'item_name'
        )
        self.trace.price_count = sum(
            1 for c in classified if c.label.value == 'item_price'
        )
    
    def record_matches(self, matches: dict[int, int], items: list, prices: list):
        """Record price-item matches."""
        self.trace.match_count = len(matches)
        self.trace.matches = [
            {
                'item_index': i,
                'price_index': j,
                'item_text': items[i].text if i < len(items) else '',
                'price_text': prices[j].text if j < len(prices) else ''
            }
            for i, j in matches.items()
        ]
    
    def finalize(self) -> PipelineTrace:
        """Finalize trace and return."""
        self.trace.total_ms = (time.time() - self._total_start) * 1000
        return self.trace


class StageTimer:
    """Context manager for timing pipeline stages."""
    
    def __init__(self, instrumentor: PipelineInstrumentor, stage_name: str):
        self.instrumentor = instrumentor
        self.stage_name = stage_name
        self.input_count = 0
        self.output_count = 0
    
    def __enter__(self):
        self.instrumentor.start_stage(self.stage_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.instrumentor.end_stage(
            self.stage_name,
            self.input_count,
            self.output_count
        )
