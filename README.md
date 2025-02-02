# Wood Defect Segmentation Network (WDSN)

## 📌 Overview

Wood Defect Segmentation Network (WDSN) is a deep learning-based solution for detecting and segmenting defects in wood surfaces. It leverages a **modified UNet** architecture with a **ResNet50** backbone and a **multi-scale decoder structure** to improve segmentation accuracy. This project compares **UNet, ResUNet, Attention UNet, and WDSN** to determine the most effective model for wood defect detection.

🔗 **Conference Paper:** [ICECET 2025](https://github.com/MJHossainSNMBU/WoodDefectsSegmentationNetword)

---

## 📂 Dataset

- The dataset consists of **5,459 images** with two segmentation classes:
  - **Defect Areas** (Live knots, Dead knots, Cracks, Resin, Marrow, etc.)
  - **Clear Wood**
- **Splits:**
  - Training: **4,365 images** (80%)
  - Validation: **547 images** (10%)
  - Testing: **547 images** (10%)
- Images are resized to **512 × 512 pixels**.

---

## 🛠 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/MJHossainSNMBU/WoodDefectsSegmentationNetword.git
cd WoodDefectsSegmentationNetword
pip install -r requirements.txt
