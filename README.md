# Menu OCR

A modular pipeline for extracting structured JSON data from restaurant menu images using OCR and multiple classification approaches.

## Features

- GPU-accelerated OCR processing
- Multiple classification approaches (rule-based, ML, ensemble)
- Spatial-aware price-item matching
- Schema-compliant JSON output
- Traceable bounding boxes for all extracted elements

## Architecture

```
Input Image → OCR Engine → Feature Extraction → Classification → Grouping → JSON Output
```

The pipeline consists of three main stages:

1. **Text Extraction**: OCR with text normalization
2. **Classification**: Rule-based, ML, or ensemble classification
3. **Hierarchical Grouping**: Spatial matching of items and prices

## Classification Approaches

### Comparison Results

| Approach | Precision | Recall | F1 | Price Accuracy |
|----------|-----------|--------|-----|----------------|
| **Rule-Based** | 41.3% | 31.8% | **35.6%** | 38.9% |
| Random Forest | 34.6% | 20.9% | 25.6% | 15.9% |
| Gradient Boosting | 34.9% | 20.9% | 25.3% | 15.9% |
| XGBoost | 37.8% | 22.5% | 27.7% | 13.6% |
| MLP Neural Net | 43.9% | 28.3% | 34.0% | 24.2% |
| Ensemble | 50.4% | 27.5% | 35.1% | **47.2%** |

### Key Finding

Rule-based classification outperforms ML models trained on receipt datasets (CORD-v2) due to domain mismatch:
- Receipts have linear layouts; menus have multi-column layouts
- Different label semantics (receipt totals vs menu items)
- Different price presentation patterns

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### GPU Support

```bash
# Verify CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Usage

### Python API

```python
from src.pipeline import MenuPipeline, PipelineConfig

config = PipelineConfig(use_gpu=True)
pipeline = MenuPipeline(config)

result = pipeline.process("menu_image.jpg")
print(result.to_json())
```

### Command Line

```bash
python main.py image.jpg -o output.json --gpu
```

## Output Schema

```json
{
  "menu": [
    {
      "id": "section_id",
      "label": "Section Name",
      "groups": [
        {
          "id": "group_id",
          "label": "Group Name",
          "items": [
            {
              "name": "Item Name",
              "price": 100.0,
              "description": "Optional description"
            }
          ]
        }
      ]
    }
  ]
}
```

## Training

Train classifiers on labeled document datasets:

```bash
# Download dataset
python scripts/download_cord.py

# Train models
python scripts/train_classifier.py --model random_forest
python scripts/train_classifier.py --model xgboost
python scripts/train_classifier.py --model mlp

# Evaluate
python scripts/evaluate_all_models.py
```

## Classification Labels

| Label | Description |
|-------|-------------|
| SECTION_HEADER | Top-level category |
| GROUP_HEADER | Sub-category |
| ITEM_NAME | Menu item name |
| ITEM_PRICE | Price value |
| ITEM_DESCRIPTION | Item description |

## Performance

- Processing time: ~330ms per image (GPU)
- GPU speedup: 3.5×
- Supported formats: JPG, PNG, TIFF

## Citation

If you use this work, please cite:

```bibtex
@misc{menuocr2024,
  title={Menu OCR: A Modular Pipeline for Structured Menu Extraction},
  author={Menu OCR Research Team},
  year={2024}
}
```

## License
MIT License © 2026 Anish Giri
