# Menu OCR

Extract structured JSON data from restaurant menu images using OCR and rule-based classification.

## Features

- **EasyOCR Backend**: Reliable text extraction with CRAFT detector + CRNN recognizer
- **Rule-based Classification**: Deterministic labeling of section headers, items, prices
- **Spatial-aware Matching**: Price-item association using bounding box proximity
- **Schema-compliant Output**: Clean JSON with sections, groups, and items
- **Web Interface**: React frontend with image upload and visualization
- **Bounding Box Visualization**: Annotated images showing detection results

## Architecture

```
┌─────────────┐    ┌────────────────┐    ┌──────────────┐    ┌─────────────┐
│  Image      │───▶│  OCR Engine    │───▶│  Classifier  │───▶│  Grouping   │
│  Input      │    │  (EasyOCR)     │    │  (Rules/ML)  │    │  (Spatial)  │
└─────────────┘    └────────────────┘    └──────────────┘    └─────────────┘
                                                                    │
                                                                    ▼
                                                            ┌─────────────┐
                                                            │  Menu JSON  │
                                                            └─────────────┘
```

## Installation

```bash
# Clone the repository
cd menu-ocr

# Install Python dependencies
uv pip install -e .

# Or with pip
pip install -r requirements.txt
```

## Quick Start

### Command Line

```python
from src.pipeline import extract_menu

# Extract menu from image
result = extract_menu(
    "path/to/menu.jpg",
    output_json="output/menu.json",
    output_image="output/annotated.jpg"
)

print(result)
```

### Python API

```python
from src.pipeline import MenuPipeline, PipelineConfig

# Configure
config = PipelineConfig(
    use_gpu=False,
    confidence_threshold=0.3
)

# Initialize
pipeline = MenuPipeline(config)

# Process
result = pipeline.process("menu.jpg")

# Get JSON
print(result.to_json())

# Visualize
pipeline.visualize("menu.jpg", result, "annotated.jpg")
```

### Web Application

```bash
# Start backend
cd web/backend
npm install
npm start  # Runs on port 3001

# Start frontend
cd web/frontend
npm install
npm run dev  # Runs on port 3000
```

## Output Schema

```json
{
  "menu": [
    {
      "id": "wines",
      "label": "Wines",
      "groups": [
        {
          "id": "imported",
          "label": "Imported",
          "items": [
            {
              "name": "Cabernet Shiraz",
              "price": 3000,
              "description": "750ml bottle"
            }
          ]
        }
      ]
    }
  ]
}
```

## Classification Labels

| Label | Description | Example |
|-------|-------------|---------|
| `section_header` | Top-level category | "WINES", "COCKTAILS" |
| `group_header` | Sub-category | "Imported", "Domestic" |
| `item_name` | Menu item name | "Margherita Pizza" |
| `item_price` | Price value | "350", "$12.99" |
| `item_description` | Item description | "with fresh basil" |

## Evaluation

Run evaluation on sample data:

```bash
python scripts/evaluate.py
```

Current Results:
- **Precision**: 55.3%
- **Recall**: 32.9%
- **F1 Score**: 40.2%
- **Price Accuracy**: 45.0%
- **Processing Time**: ~2.2s per image

## Project Structure

```
menu-ocr/
├── src/
│   ├── models/
│   │   └── schema.py      # Pydantic data models
│   ├── ocr/
│   │   └── engine.py      # EasyOCR wrapper
│   ├── classifier/
│   │   └── classifier.py  # Rule-based + ML classifier
│   └── pipeline.py        # Main extraction pipeline
├── web/
│   ├── backend/           # Express API server
│   └── frontend/          # React application
├── scripts/
│   ├── evaluate.py        # Evaluation script
│   └── api_extract.py     # API extraction helper
├── data/
│   └── samples/           # Test samples with ground truth
├── output/                # Generated outputs
└── paper/                 # LaTeX paper
```

## License

MIT
