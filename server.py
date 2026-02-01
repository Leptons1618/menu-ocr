#!/usr/bin/env python3
"""
FastAPI inference server for Menu OCR.
Supports multiple models and ensemble inference.
"""

import io
import time
import json
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2

from src.models.schema import TextElementType
from src.ocr.engine import OCREngine
from src.classifier.classifier import MenuClassifier
from src.classifier.ensemble import EnsembleClassifier
from src.pipeline import MenuPipeline, PipelineConfig, PipelineResult

# Directories
UPLOAD_DIR = Path("web/uploads")
OUTPUT_DIR = Path("web/output")
MODELS_DIR = Path("models")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Available models cache
class ModelCache:
    """Cache for loaded models."""
    
    def __init__(self):
        self.pipelines = {}
        self.available_models = []
        self.ensemble = None
    
    def scan_models(self):
        """Scan for available model files."""
        self.available_models = ["rule_based"]  # Always available
        
        for pkl_file in MODELS_DIR.glob("*.pkl"):
            self.available_models.append(pkl_file.stem)
        
        if len(self.available_models) > 2:
            self.available_models.append("ensemble")
        
        return self.available_models
    
    def get_pipeline(self, model_name: str, use_gpu: bool = True) -> MenuPipeline:
        """Get or create pipeline for model."""
        cache_key = f"{model_name}_{use_gpu}"
        
        if cache_key not in self.pipelines:
            if model_name == "rule_based":
                config = PipelineConfig(use_gpu=use_gpu, model_path=None)
                self.pipelines[cache_key] = MenuPipeline(config)
            
            elif model_name == "ensemble":
                # Load all available models for ensemble
                model_paths = list(MODELS_DIR.glob("*.pkl"))
                self.pipelines[cache_key] = EnsemblePipeline(
                    model_paths=model_paths,
                    use_gpu=use_gpu
                )
            
            else:
                model_path = MODELS_DIR / f"{model_name}.pkl"
                if not model_path.exists():
                    raise ValueError(f"Model not found: {model_name}")
                config = PipelineConfig(use_gpu=use_gpu, model_path=model_path)
                self.pipelines[cache_key] = MenuPipeline(config)
        
        return self.pipelines[cache_key]


class EnsemblePipeline:
    """Pipeline using ensemble classifier."""
    
    def __init__(self, model_paths: List[Path], use_gpu: bool = True):
        self.ocr = OCREngine(use_gpu=use_gpu)
        self.classifier = EnsembleClassifier(
            model_paths=model_paths,
            use_rules=True,
            rule_weight=1.5
        )
        self.config = PipelineConfig(use_gpu=use_gpu)
    
    def process(self, image) -> PipelineResult:
        """Process image with ensemble."""
        from src.pipeline import slugify, extract_price
        from src.models.schema import MenuDocument, MenuSection, MenuGroup, MenuItem
        import re
        
        start = time.time()
        
        # Get image dimensions
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                img_w, img_h = img.size
        elif isinstance(image, Image.Image):
            img_w, img_h = image.size
        else:
            img_h, img_w = image.shape[:2]
        
        # OCR
        ocr_results = self.ocr.extract(image, confidence_threshold=0.3)
        
        if not ocr_results:
            return PipelineResult(
                document=MenuDocument(menu=[]),
                warnings=["No text detected"]
            )
        
        # Classify with ensemble
        classified = self.classifier.classify_all(ocr_results, img_w, img_h)
        
        # Build menu structure (same as MenuPipeline)
        sections = []
        current_section = None
        current_group = None
        
        for ct in classified:
            label = ct.label
            text = ct.text.strip()
            
            if label == TextElementType.SECTION_HEADER:
                if current_group and current_group.items:
                    if current_section:
                        current_section.groups.append(current_group)
                if current_section and current_section.groups:
                    sections.append(current_section)
                
                current_section = MenuSection(id=slugify(text), label=text)
                current_group = None
            
            elif label == TextElementType.GROUP_HEADER:
                if current_group and current_group.items:
                    if current_section:
                        current_section.groups.append(current_group)
                current_group = MenuGroup(id=slugify(text), label=text)
            
            elif label == TextElementType.ITEM_NAME:
                if not current_group:
                    current_group = MenuGroup(id="default", label=None)
                if not current_section:
                    current_section = MenuSection(id="menu", label="Menu")
                
                # Extract price (check if embedded in text)
                price = None
                match = re.search(r'\s+([\d,]+(?:\.\d{2})?)\s*$', text)
                if match:
                    price = extract_price(match.group(1))
                    text = text[:match.start()].strip()
                
                current_group.items.append(MenuItem(name=text, price=price))
        
        if current_group and current_group.items:
            if current_section:
                current_section.groups.append(current_group)
        if current_section and current_section.groups:
            sections.append(current_section)
        
        return PipelineResult(
            document=MenuDocument(menu=sections),
            ocr_results=ocr_results,
            classified=classified,
            processing_time_ms=(time.time() - start) * 1000
        )
    
    def visualize(self, image_path, result, output_path):
        """Visualize results."""
        img = cv2.imread(str(image_path))
        
        colors = {
            TextElementType.SECTION_HEADER: (0, 0, 255),
            TextElementType.GROUP_HEADER: (0, 165, 255),
            TextElementType.ITEM_NAME: (0, 255, 0),
            TextElementType.ITEM_PRICE: (255, 0, 0),
            TextElementType.ITEM_DESCRIPTION: (255, 255, 0),
            TextElementType.METADATA: (128, 128, 128),
            TextElementType.OTHER: (200, 200, 200),
        }
        
        for ct in result.classified:
            bbox = ct.bbox
            color = colors.get(ct.label, (200, 200, 200))
            pt1 = (int(bbox.x_min), int(bbox.y_min))
            pt2 = (int(bbox.x_max), int(bbox.y_max))
            cv2.rectangle(img, pt1, pt2, color, 2)
        
        cv2.imwrite(str(output_path), img)
        return img


# Global model cache
model_cache = ModelCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    # Startup: scan models
    models = model_cache.scan_models()
    print(f"Available models: {models}")
    yield
    # Shutdown
    print("Shutting down...")


# Create app
app = FastAPI(
    title="Menu OCR API",
    description="Extract structured data from menu images",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


# Response models
class HealthResponse(BaseModel):
    status: str
    models: List[str]
    gpu_available: bool


class ExtractResponse(BaseModel):
    success: bool
    menu: dict
    images: dict
    processing_time_ms: float
    model_used: str


# Endpoints
@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check and available models."""
    import torch
    return {
        "status": "ok",
        "models": model_cache.scan_models(),
        "gpu_available": torch.cuda.is_available()
    }


@app.get("/api/models")
async def list_models():
    """List available models."""
    models = model_cache.scan_models()
    
    model_info = []
    for name in models:
        info = {"name": name, "type": "rule_based" if name == "rule_based" else "ml"}
        
        # Load metadata if available
        json_path = MODELS_DIR / f"{name}.json"
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
                info.update(meta)
        
        model_info.append(info)
    
    return {"models": model_info}


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(
    image: UploadFile = File(...),
    model: str = Query(default="rule_based", description="Model to use"),
    use_gpu: bool = Query(default=True, description="Use GPU acceleration")
):
    """Extract menu from image."""
    
    # Validate model
    available = model_cache.scan_models()
    if model not in available:
        raise HTTPException(400, f"Unknown model: {model}. Available: {available}")
    
    # Save uploaded file
    timestamp = int(time.time() * 1000)
    filename = f"{timestamp}_{image.filename}"
    image_path = UPLOAD_DIR / filename
    
    contents = await image.read()
    with open(image_path, 'wb') as f:
        f.write(contents)
    
    try:
        # Get pipeline
        pipeline = model_cache.get_pipeline(model, use_gpu)
        
        # Process
        result = pipeline.process(str(image_path))
        
        # Save outputs
        basename = image_path.stem
        output_json = OUTPUT_DIR / f"{basename}.json"
        output_image = OUTPUT_DIR / f"{basename}_annotated.jpg"
        
        with open(output_json, 'w') as f:
            f.write(result.to_json())
        
        pipeline.visualize(str(image_path), result, str(output_image))
        
        return {
            "success": True,
            "menu": result.to_dict(),
            "images": {
                "original": f"/uploads/{filename}",
                "annotated": f"/output/{basename}_annotated.jpg"
            },
            "processing_time_ms": result.processing_time_ms,
            "model_used": model
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/compare")
async def compare_models(image_path: str = Query(...)):
    """Compare results from all models on same image."""
    
    if not Path(image_path).exists():
        raise HTTPException(404, "Image not found")
    
    results = {}
    for model_name in model_cache.scan_models():
        try:
            pipeline = model_cache.get_pipeline(model_name, use_gpu=True)
            result = pipeline.process(image_path)
            
            # Count items
            item_count = sum(
                len(g.items) 
                for s in result.document.menu 
                for g in s.groups
            )
            
            results[model_name] = {
                "items_extracted": item_count,
                "processing_time_ms": result.processing_time_ms,
                "ocr_detections": len(result.ocr_results)
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}
    
    return {"comparisons": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
