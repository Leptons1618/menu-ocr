# Menu OCR

A modular pipeline for extracting structured JSON data from restaurant menu images using OCR and multiple classification approaches.

## Features

- GPU-accelerated OCR processing
- **Hybrid classification** (rule-based primary, ML secondary)
- **Column-aware layout detection** using DBSCAN clustering
- **Global bipartite matching** for price-item association (Hungarian algorithm)
- **Hierarchy enforcement** via finite-state machine with Viterbi decoding
- **Lexical priors** for menu-domain terms
- Schema-compliant JSON output
- **Pipeline instrumentation** with timing and diagnostics

## Architecture

```
Input Image → OCR → Column Detection → Classification → Hierarchy FSM → Bipartite Matching → JSON
                         ↓                    ↓
                   Price Column         Lexical Priors
                   Detection            Font Analysis
```

The pipeline consists of six main stages:

1. **Text Extraction**: OCR with extended spatial metadata
2. **Column Detection**: DBSCAN x-coordinate clustering, price column identification
3. **Classification**: Hybrid rule-based + ML with lexical priors
4. **Hierarchy Enforcement**: Viterbi decoding with FSM transition constraints
5. **Price Matching**: Hungarian algorithm bipartite matching
6. **Grouping**: Build hierarchical menu structure

## Key Improvements (v2.0)

### Column-Aware Layout Segmentation
- Explicit column detection via x-coordinate clustering
- Automatic price column identification (rightmost, high digit ratio)
- Multi-column reading order resolution

### Global Price-Item Matching
- Replaced greedy matching with Hungarian algorithm
- Cost function considers: vertical alignment, horizontal penalty, column consistency
- Prevents conflicts where multiple items claim same price

### Hierarchy Detection
- Finite-state machine enforcing valid transitions (Section→Group→Item→Price)
- Viterbi decoding for globally optimal label sequence
- Font scale analysis using KMeans clustering
- Lexical priors for section/group keywords

### ML as Secondary Signal
- Rule-based classification as primary (addresses receipt-menu domain mismatch)
- ML models only adjust predictions when rule confidence < 0.7

## Classification Results

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

**Recommendation**: Use receipt-trained ML models as secondary signals only.

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

# Full feature configuration
config = PipelineConfig(
    use_gpu=True,
    use_column_detection=True,
    use_bipartite_matching=True,
    use_hierarchy_fsm=True,
    use_lexical_priors=True,
    save_diagnostics=True
)
pipeline = MenuPipeline(config)

result = pipeline.process("menu_image.jpg")
print(result.to_json())

# Access diagnostics
print(f"Columns detected: {len(result.columns)}")
print(f"Matches: {result.matches}")
print(f"Processing time: {result.trace.total_ms}ms")
```

### Command Line

```bash
python main.py image.jpg -o output.json --gpu
```

### Diagnostic Output

Enable `save_diagnostics=True` to generate per-stage visualizations:
- `detections.png` - OCR bounding boxes
- `columns.png` - Detected column boundaries
- `classifications.png` - Color-coded label assignments
- `matches.png` - Item-price matching lines

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

# Error analysis with confusion matrices
python scripts/error_analysis.py --output-dir output/analysis
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
