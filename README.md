# Lung Nodule Detection — Deep Learning Pipeline

A complete deep learning pipeline for automated lung nodule detection and segmentation in chest CT scans, combining U-Net segmentation, GAN-based data augmentation, and false positive reduction.

**Results:** F1-score = 0.84 | IoU = 0.79 | Dataset: 10,000+ chest radiographs

---

## Project Structure

```
├── U-net segmentation/
│   ├── Train_Unet.py          # Core model: U-Net architecture + training loop
│   ├── Model_Prediction.py    # Inference and prediction on new scans
│   ├── Data_Augmentation.py   # Image augmentation pipeline
│   ├── Dicom_2_Png.py         # DICOM to PNG conversion
│   ├── Get_mask.py            # Ground truth mask extraction
│   ├── Image_resize.py        # Preprocessing / resizing
│   └── data/Test_images/      # Sample test images
│
├── GAN date augmentation/
│   ├── 2A_train_injector.py   # Train GAN for nodule injection
│   ├── 2B_train_remover.py    # Train GAN for nodule removal
│   ├── 3A_inject_evidence.py  # Inject synthetic nodules into scans
│   ├── 3B_remove evidence.py  # Remove nodules from scans
│   ├── config.py              # Configuration (paths, GAN settings)
│   ├── procedures/trainer.py  # GAN training logic
│   └── utils/                 # Dataloader, DICOM utils, equalizer
│
└── Reduce false positive/
    ├── model.py               # MGICNN: Multi-scale false positive reducer
    ├── main.py                # Training and evaluation entry point
    ├── layers.py              # Custom layer definitions
    ├── plot_roc.py            # ROC curve visualization
    └── settings.py            # Hyperparameters and config
```

---

## Pipeline Overview

```
CT Scans (DICOM)
      │
      ▼
[1] Preprocessing
    Dicom_2_Png.py + Image_resize.py
      │
      ▼
[2] GAN Data Augmentation
    Synthetic nodule injection/removal to balance the dataset
      │
      ▼
[3] U-Net Segmentation  ← core model
    Train_Unet.py
    Input: 320×320 grayscale CT slices
    Output: binary segmentation mask
      │
      ▼
[4] False Positive Reduction
    MGICNN (model.py) — multi-scale 3-slice CNN
    Filters out anatomical structures misclassified as nodules
      │
      ▼
[5] Evaluation
    F1 = 0.84 | IoU = 0.79
```

---

## Model Architecture

### U-Net (Train_Unet.py)

An encoder-decoder convolutional network with skip connections for pixel-level segmentation:

- **Input:** 320×320 × 1 (grayscale CT slice)
- **Encoder:** 5 blocks of Conv2D (3×3, ReLU) + MaxPooling, filters: 32 → 64 → 128 → 256 → 256
- **Decoder:** UpSampling2D + concatenation (skip connections) + Conv2D blocks
- **Output:** 320×320 × 1 sigmoid mask
- **Loss:** Dice coefficient loss
- **Optimizer:** SGD (lr=0.001, momentum=0.9, Nesterov)

```python
# Dice loss function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 100) / (K.sum(y_true_f) + K.sum(y_pred_f) + 100)
```

### MGICNN — False Positive Reduction (model.py)

Multi-scale Gradual Integration CNN operating on 3 consecutive CT slices (bottom, middle, top):

- Two parallel GFE (Gradual Feature Extraction) streams: zoom-in and zoom-out
- Streams fused by element-wise addition or concatenation
- Final layers: Flatten → Dense(1024) × 2 → sigmoid output
- **Optimizer:** Adam | **Loss:** sigmoid cross-entropy

### CT-GAN Data Augmentation

Conditional GAN for injecting and removing nodule evidence in 3D CT volumes, used to address class imbalance in training data.

---

## Data Augmentation (Training)

The `image_generator()` in `Train_Unet.py` applies on-the-fly augmentation:

| Technique | Parameters |
|---|---|
| Elastic deformation | alpha=128, sigma=15 |
| Random rotation | ±20°, p=0.8 |
| Random flip | horizontal + vertical, p=0.5 |
| Random translation | ±30px, p=0.8 |
| Brightness / Contrast / Color jitter | range [0.5, 1.5] |

---

## Error Analysis

**Main source of false positives:** Blood vessels and rib cross-sections share similar circular shapes and intensity profiles with small nodules on 2D CT slices.

**Diagnosis:** The U-Net alone achieved high IoU (0.79) but produced false positives on vascular structures.

**Fix applied:** The MGICNN false positive reducer uses 3 consecutive slices to leverage 3D context — blood vessels appear tubular across slices while true nodules appear as isolated spherical masses. This significantly reduced the false positive rate.

---

## Training

```bash
# 1. Edit paths in Train_Unet.py
TRAIN_LIST = './data/chapter4/train_img.txt'
VAL_LIST   = './data/chapter4/val_img.txt'

# 2. Train U-Net from scratch
python "U-net segmentation/Train_Unet.py"

# 3. Train false positive reducer
python "Reduce false positive/main.py"

# 4. (Optional) Run GAN augmentation
python "GAN date augmentation/2A_train_injector.py"
```

**Training config (U-Net):**
- Batch size: 8 | Epochs: 80 | Steps per epoch: 200
- Model saved via `ModelCheckpoint` (best val_loss)

---

## Requirements

```bash
pip install tensorflow keras opencv-python numpy scipy SimpleITK Pillow matplotlib
```

- Python 3.6+
- Keras with TensorFlow backend (≥ 1.13)
- GPU recommended

---

## Results

| Metric | Score |
|---|---|
| F1-score | **0.84** |
| IoU | **0.79** |
| Dataset size | 10,000+ CT slices |

---

## Author

**Amani Brahimi** — [@amanibrahimi](https://github.com/amanibrahimi)  
Master 1 Big Data Analytics — USTHB, Alger  
1597amani@gmail.com
