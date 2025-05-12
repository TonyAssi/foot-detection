# 🦶 FootDetection

A lightweight Python module for detecting feet or shoes in images using a fine-tuned Faster R-CNN model (PyTorch + Torchvision).

Trained on a small custom dataset and supports both CPU and Apple Silicon GPU (MPS) inference.  
Model weights are automatically downloaded from Hugging Face.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/tonyassi/foot-detection)

---

## 🚀 Features

- 🧠 Fine-tuned `fasterrcnn_resnet50_fpn` from Torchvision
- 🦶 Detects feet and shoes with bounding boxes and confidence scores
- 🪶 Clean class-based API
- ✅ Works on CPU or Apple M1/M2 GPU (`mps`)
- 🔗 Automatically downloads weights from Hugging Face on first run

---

## ⬇️ Download

To download this repository:

```bash
git clone https://github.com/tonyassi/FootDetection.git
cd FootDetection
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🖼️ Usage

```python
from FootDetection import FootDetection
from PIL import Image

# Initialize model (first run will auto-download weights)
foot_detection = FootDetection("cpu")  # "cuda" for GPU  or "mps" for Apple Silicon

# Load image
img = Image.open("image.jpg").convert("RGB")

# Run detection
results = foot_detection.detect(img, threshold=0.1)
print(results)

# Draw boxes
img_with_boxes = foot_detection.draw_boxes(img)
img_with_boxes.show()
img_with_boxes.save("annotated_image.jpg")
```
