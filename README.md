# 🎯 Military Asset Detection using YOLOv8

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0%2Bcu126-EE4C2C.svg?logo=pytorch)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3.235-black.svg)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end automated pipeline for training, evaluating, and packaging a YOLOv8 object detection model to identify 12 classes of military assets. This repository contains the Colab notebook and configurations used to process the dataset, train a YOLOv8m model, and generate prediction artifacts for deployment or submission.

---

## 📖 Project Overview

This project builds a custom object detector capable of identifying military vehicles, personnel, and infrastructure from images. The pipeline is designed to run in Google Colab (T4 GPU optimized) and handles everything from automated dataset extraction and validation to inference on test data and formatting outputs for submission.

## 📊 Dataset & Class Distribution

The model is trained on a custom 12-class `military_object_dataset`. 

⚠️ **Note on Dataset Imbalance:** The current dataset exhibits severe class imbalance, which significantly impacts model recall on minority classes.

| Dominant Classes | Instances | Critically Starved Classes | Instances |
| :--- | :--- | :--- | :--- |
| **Military Tank** | ~7,800 | **Civilian** | 1 |
| **Military Aircraft** | ~3,500 | **Trench** | 1 |
| **Soldier** | ~3,000 | **Military Warship** | 0 |

## ⚙️ Model & Training Configuration

The project utilizes a pretrained **YOLOv8 Medium (YOLOv8m)** model to strike an optimal balance between inference speed and bounding box accuracy. 

* **Image Size:** 640x640
* **Epochs:** 40 (with Early Stopping patience of 20)
* **Batch Size:** 16
* **Optimizer:** AdamW (helps regularize the imbalanced learning)
* **Compute:** T4 GPU with Automatic Mixed Precision (AMP) enabled for faster training.

## 📈 Performance Metrics

Due to the heavy dataset imbalance, the model currently experiences a "Recall Bottleneck". 

* **Precision:** `0.69` (High confidence when an object is detected).
* **Recall:** `0.42` (High rate of false negatives for starved classes).

*Future iterations will implement focal loss, heavy mosaic augmentations, and synthetic data generation to address the minority classes.*

## 🚀 Pipeline Execution

The provided Jupyter/Colab notebook executes the following workflow natively:

1. **Environment Setup:** Mounts Google Drive, unzips data, and installs dependencies (`ultralytics`, `opencv-python-headless`).
2. **Data Validation:** Verifies `train`, `val`, and `test` structural integrity and plots bounding-box area histograms.
3. **Configuration:** Auto-generates `military_dataset.yaml`.
4. **Training:** Executes the YOLOv8m training loop.
5. **Validation:** Runs a standalone validation pass on `best.pt` calculating mAP50.
6. **Inference & Packaging:** 
   - Runs inference on 1,396 unseen test images.
   - Outputs standard YOLO formatted `.txt` files (`<class_index> <xc> <yc> <w> <h> <conf>`).
   - Packages all predictions into `submission.zip`.
   - Auto-downloads `submission.zip` and `best.pt` to the local machine.

## 🛠️ Requirements & Setup

To run this pipeline locally or in your own Colab environment, ensure you have the following installed:

```bash
pip install ultralytics==8.3.235
pip install opencv-python-headless
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Ensure your zipped dataset is named `military_object_dataset.zip` and structured with standard YOLO `images/` and `labels/` subdirectories for `train`, `val`, and `test` splits.

## 🤝 Contributing & Future Work

Contributions to improve model recall are highly encouraged! Immediate areas for collaboration include:
* Implementing Class Weights / Focal Loss.
* Sourcing or generating synthetic data for `trench`, `civilian`, and `military_warship` classes.
* Fine-tuning data augmentation hyperparameters (MixUp, Mosaic, HSV adjustments).

---
*Created with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)*
