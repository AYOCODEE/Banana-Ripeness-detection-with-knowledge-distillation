# Banana Ripeness Detection with Knowledge Distillation

[![CI](https://github.com/AYOCODEE/Banana-Ripeness-detection-with-knowledge-distillation/actions/workflows/ci.yml/badge.svg)](https://github.com/AYOCODEE/Banana-Ripeness-detection-with-knowledge-distillation/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A deep-learning pipeline that classifies banana ripeness into four categories
> using knowledge distillation — compressing a large **ResNet50 teacher** model
> into a lightweight **custom ResNet10 student** model.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Results](#results)
- [API](#api)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

Banana ripeness detection is important for quality control in food supply chains
and retail. This project:

1. **Trains a teacher model** (ResNet50 pretrained on ImageNet) to classify
   bananas into four ripeness levels.
2. **Distils knowledge** from the teacher into a lightweight custom ResNet10
   student model using Hinton's Knowledge Distillation technique.
3. **Serves predictions** via a FastAPI REST API for integration into
   downstream applications.

### Ripeness Classes

| Label | Description |
|-------|-------------|
| Unripe | Green, not ready to eat |
| Ripe | Yellow, optimal for consumption |
| Overripe | Brown spots, very sweet |
| Rotten | Deteriorated, not suitable for consumption |

---

## Model Architecture

### Teacher — ResNet50

- Pretrained on ImageNet (frozen backbone)
- Custom head: `GlobalAveragePooling2D → Dense(512, relu) → Dropout(0.4) → Dense(256, relu) → Dropout(0.4) → Dense(4)`

### Student — Custom ResNet10

- Lightweight custom residual network with 4 basic blocks
- Channels: 64 → 128 → 192 → 256
- `GlobalAveragePooling2D → Dense(4)`

### Knowledge Distillation

- **Alpha** (task loss weight): 0.1
- **Temperature**: 40
- **Distillation loss**: KL Divergence on softened logits

---

## Dataset

The project uses the
[Banana Ripeness Classification dataset](https://universe.roboflow.com/)
(augmented 3× with modified classes, 4-class folder format).

Expected directory structure:

```
data/raw/
├── Overripe/
├── Ripe/
├── Unripe/
└── Rotten/
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# Clone the repository
git clone https://github.com/AYOCODEE/Banana-Ripeness-detection-with-knowledge-distillation.git
cd Banana-Ripeness-detection-with-knowledge-distillation

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

For development (linting, testing):

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

```python
from src.inference import predict_from_path

result = predict_from_path(
    model_path="models/resnet10_distilled_student_model",
    image_path="path/to/banana.jpg",
)
print(result["predicted_class"])  # e.g. "Ripe"
print(f"Confidence: {result['confidence']:.2%}")
```

---

## Usage

### Command-Line Inference

```bash
python -m src.inference \
    --model models/resnet10_distilled_student_model \
    --image path/to/banana.jpg
```

### Training

See the notebooks in `notebooks/` for the full training walkthrough, or run
the relevant sections from `src/models/`.

### Running Tests

```bash
pytest tests/ -v
```

---

## Results

| Model | Parameters | Test Accuracy |
|-------|-----------|--------------|
| Teacher (ResNet50) | ~25 M | ~XX% |
| Student (ResNet10) with distillation | ~3 M | ~XX% |
| Student (ResNet10) without distillation | ~3 M | ~XX% |

> Actual accuracy figures depend on the dataset split and training run.

---

## API

Start the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/classes` | List ripeness classes |
| POST | `/predict` | Upload an image and get a prediction |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@banana.jpg"
```

**Example response:**

```json
{
  "predicted_class": "Ripe",
  "confidence": 0.93,
  "probabilities": {
    "Overripe": 0.03,
    "Ripe": 0.93,
    "Unripe": 0.02,
    "Rotten": 0.02
  }
}
```

Interactive docs are available at `http://localhost:8000/docs`.

---

## Docker

```bash
# Build the image
docker build -t banana-ripeness .

# Run the container
docker run -p 8000:8000 -v $(pwd)/models:/app/models banana-ripeness
```

---

## Project Structure

```
├── app.py                   # FastAPI REST API
├── configs/
│   ├── config.yaml          # Training hyperparameters
│   └── inference.yaml       # Inference settings
├── Dockerfile
├── notebooks/               # Original Jupyter notebooks
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── pyproject.toml
├── src/
│   ├── data/
│   │   └── dataloader.py    # Data loading & augmentation
│   ├── models/
│   │   ├── distiller.py     # Knowledge distillation logic
│   │   ├── student.py       # Custom ResNet10 student
│   │   └── teacher.py       # ResNet50 teacher
│   ├── utils/
│   │   ├── metrics.py       # Evaluation metrics
│   │   └── visualization.py # Plotting utilities
│   └── inference.py         # Inference pipeline
└── tests/                   # pytest unit tests
```

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details.
