# RSNA Pneumonia Detection README

This repository implements a pipeline for training a Faster R-CNN model on the RSNA Pneumonia Detection dataset. The system detects pneumonia in chest X-ray images using bounding boxes.

---

## Features

- **Data Handling**:
  - Reads and processes RSNA dataset with bounding box annotations.
  - Splits data into training and validation subsets.
  - Implements data augmentation and transformations.

- **Model**:
  - Utilizes a pre-trained Faster R-CNN with ResNet50 backbone.
  - Customizes the classifier head for binary classification (pneumonia vs. no pneumonia).

- **Metrics**:
  - Implements precision calculation with Intersection over Union (IoU).
  - Includes utility functions for evaluation and visualization.

- **Training**:
  - Distributed training using the `Accelerate` library.
  - Includes support for logging metrics via Tensorboard and CometML.

- **Visualization**:
  - Saves input images with ground truth and predicted bounding boxes.

---

## Prerequisites

### Dependencies
Ensure the following Python packages are installed:
- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `cv2`
- `pydicom`
- `Pillow`
- `argparse`
- `accelerate`

Install them with:

```bash
pip install torch torchvision numpy pandas matplotlib opencv-python pydicom pillow accelerate
```

---

## Usage

To train the model, run:

```bash
python train.py --data_dir path/to/data --epochs 50 --batch_size 8
```

To evaluate the model, run:

```bash
python evaluate.py --data_dir path/to/data --checkpoint path/to/checkpoint
```

---

## Acknowledgements

This project is based on the RSNA Pneumonia Detection Challenge dataset. Special thanks to the RSNA and the contributors of the dataset.
