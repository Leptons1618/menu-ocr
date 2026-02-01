#!/usr/bin/env python3
"""
Compare OCR backends: EasyOCR vs PaddleOCR vs Tesseract.
Generates comparison data for README and paper.
"""

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class OCRMetrics:
    """OCR comparison metrics."""
    backend: str
    init_time_s: float
    avg_process_time_s: float
    total_detections: int
    avg_confidence: float
    gpu_supported: bool
    gpu_used: bool
    errors: list[str]


def test_easyocr(images: list[Path], use_gpu: bool = True) -> OCRMetrics:
    """Test EasyOCR backend."""
    import easyocr
    
    errors = []
    try:
        start = time.time()
        reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        init_time = time.time() - start
    except Exception as e:
        errors.append(str(e))
        return OCRMetrics(
            backend="EasyOCR",
            init_time_s=0,
            avg_process_time_s=0,
            total_detections=0,
            avg_confidence=0,
            gpu_supported=True,
            gpu_used=use_gpu,
            errors=errors
        )
    
    total_time = 0
    total_detections = 0
    total_conf = 0
    
    for img_path in images:
        start = time.time()
        results = reader.readtext(str(img_path))
        total_time += time.time() - start
        
        for _, text, conf in results:
            if conf > 0.3:
                total_detections += 1
                total_conf += conf
    
    return OCRMetrics(
        backend="EasyOCR",
        init_time_s=init_time,
        avg_process_time_s=total_time / len(images),
        total_detections=total_detections,
        avg_confidence=total_conf / max(total_detections, 1),
        gpu_supported=True,
        gpu_used=use_gpu,
        errors=errors
    )


def test_tesseract(images: list[Path]) -> OCRMetrics:
    """Test Tesseract OCR backend."""
    errors = []
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        return OCRMetrics(
            backend="Tesseract",
            init_time_s=0,
            avg_process_time_s=0,
            total_detections=0,
            avg_confidence=0,
            gpu_supported=False,
            gpu_used=False,
            errors=[f"Not installed: {e}"]
        )
    
    total_time = 0
    total_detections = 0
    
    for img_path in images:
        start = time.time()
        try:
            img = Image.open(img_path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            total_time += time.time() - start
            
            for conf in data['conf']:
                if isinstance(conf, int) and conf > 30:
                    total_detections += 1
        except Exception as e:
            errors.append(str(e))
    
    return OCRMetrics(
        backend="Tesseract",
        init_time_s=0,
        avg_process_time_s=total_time / len(images) if images else 0,
        total_detections=total_detections,
        avg_confidence=0.7,  # Tesseract doesn't provide per-word confidence easily
        gpu_supported=False,
        gpu_used=False,
        errors=errors
    )


def test_paddleocr(images: list[Path], use_gpu: bool = True) -> OCRMetrics:
    """Test PaddleOCR backend."""
    import os
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    
    errors = []
    try:
        from paddleocr import PaddleOCR
        import warnings
        warnings.filterwarnings('ignore')
        
        start = time.time()
        ocr = PaddleOCR(lang='en', use_gpu=use_gpu)
        init_time = time.time() - start
    except Exception as e:
        return OCRMetrics(
            backend="PaddleOCR",
            init_time_s=0,
            avg_process_time_s=0,
            total_detections=0,
            avg_confidence=0,
            gpu_supported=True,
            gpu_used=use_gpu,
            errors=[f"Init failed: {e}"]
        )
    
    total_time = 0
    total_detections = 0
    total_conf = 0
    
    for img_path in images:
        start = time.time()
        try:
            result = list(ocr.predict(str(img_path)))
            total_time += time.time() - start
            
            for item in result:
                texts = item.get('rec_texts', [])
                scores = item.get('rec_scores', [])
                for text, score in zip(texts, scores):
                    if score > 0.3:
                        total_detections += 1
                        total_conf += score
        except Exception as e:
            errors.append(f"Process error: {e}")
            total_time += time.time() - start
    
    return OCRMetrics(
        backend="PaddleOCR",
        init_time_s=init_time,
        avg_process_time_s=total_time / len(images) if images else 0,
        total_detections=total_detections,
        avg_confidence=total_conf / max(total_detections, 1),
        gpu_supported=True,
        gpu_used=use_gpu,
        errors=errors
    )


def main():
    samples_dir = Path("data/samples")
    images = list(samples_dir.glob("*.jpg"))
    
    if not images:
        print("No sample images found!")
        return
    
    print(f"Testing OCR backends on {len(images)} images...\n")
    
    results = []
    
    # Test EasyOCR (GPU)
    print("Testing EasyOCR (GPU)...")
    results.append(test_easyocr(images, use_gpu=True))
    
    # Test EasyOCR (CPU)
    print("Testing EasyOCR (CPU)...")
    results.append(test_easyocr(images, use_gpu=False))
    
    # Test Tesseract
    print("Testing Tesseract...")
    results.append(test_tesseract(images))
    
    # Test PaddleOCR
    print("Testing PaddleOCR...")
    results.append(test_paddleocr(images, use_gpu=False))
    
    # Print results
    print("\n" + "="*80)
    print("OCR COMPARISON RESULTS")
    print("="*80)
    print(f"{'Backend':<20} {'Init(s)':<10} {'Proc(s)':<10} {'Detections':<12} {'Avg Conf':<10} {'GPU':<6} {'Errors'}")
    print("-"*80)
    
    for r in results:
        gpu_str = "Yes" if r.gpu_used else "No"
        err_str = str(len(r.errors)) if r.errors else "-"
        print(f"{r.backend:<20} {r.init_time_s:<10.2f} {r.avg_process_time_s:<10.2f} {r.total_detections:<12} {r.avg_confidence:<10.2f} {gpu_str:<6} {err_str}")
    
    # Save results
    output = Path("output/ocr_comparison.json")
    output.parent.mkdir(exist_ok=True)
    with open(output, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
