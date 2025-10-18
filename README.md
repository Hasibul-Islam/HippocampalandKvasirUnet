# 🧩 UNet Image Segmentation Project

A clean and modular PyTorch implementation of **UNet** for [Hippocampal!](https://rutgers.box.com/v/HippocampalROISegDataset) image segmentation.  And also Implementation on [Kvasir!](https://www.kaggle.com/datasets/debeshjha1/kvasirseg?resource=download). Dataset. 


---

## 🚀 Features

- Fully modular project architecture (datasets, models, utils, config, training)
- Custom **Dice Loss** for segmentation
- **Early stopping** and best-model checkpointing
- Train/validation/test pipeline with augmentations (Albumentations)
- GPU/CPU auto-selection
- Ready for experimentation and dataset extension

---

## 📁 Project Structure

```
unet_segmentation_project/
│
├── data/
│   ├── images/                 # Input images (.tif)
│   └── masks/                  # Ground truth masks (mask.png)
│
├── models/
│   └── unet.py                 # UNet architecture
│
├── datasets/
│   └── segmentation_dataset.py # Dataset class + augmentations
│
├── utils/
│   ├── losses.py               # DiceLoss and other metrics
│   ├── train_eval.py           # Training & evaluation loops
│   └── config.py               # Global constants and hyperparameters
│
├── train.py                    # Main training script
├── test.py                     # (Optional) Inference script
├── requirements.txt            # Library dependencies
└── README.md                   # Project documentation
```

---



## ⚙️ Installation Guide

### 1️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

---

### 2️⃣ Install PyTorch Nightly (CUDA 12.8)

> ⚠️ You must install PyTorch **manually** if you’re using the nightly CUDA build.

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

### 3️⃣ Install other dependencies

```bash
pip install -r requirements.txt
```

#### requirements.txt

```
# PyTorch installed manually (nightly build, CUDA 12.8)
# Install separately using:
# pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

albumentations
opencv-python
numpy
scikit-learn
tqdm
```

---

## 🧩 Dataset Setup

Organize your dataset as follows:

```
data/
│
├── images/
│   ├── sample1.tif
│   ├── sample2.tif
│   └── ...
│
└── masks/
    ├── sample1_mask.png
    ├── sample2_mask.png
    └── ...
```

> Image names must match mask names (except the `_mask` suffix).

Example:
```
image: sample1.tif
mask:  sample1_mask.png
```

---

## 🧠 Training

Run the main training scripts:

1. For Cohort1 Multiplexed
```bash
python trainCohortMul.py
```

The script will:
- Split the dataset into **train/val/test** sets (15% test, 15% val)
- Train the UNet model
- Save the **best model** as `unet_bestCohortMul.pth` using early stopping

You’ll see progress and loss updates directly in the terminal.

---

2. For Cohort1 NeuN
```bash
python trainCohortMul.py
```

The script will:
- Split the dataset into **train/val/test** sets (15% test, 15% val)
- Train the UNet model
- Save the **best model** as `unet_bestCohortNeuN.pth` using early stopping

You’ll see progress and loss updates directly in the terminal.

---

3. For Extended
```bash
python trainCohortMul.py
```

The script will:
- Split the dataset into **train/val/test** sets (15% test, 15% val)
- Train the UNet model
- Save the **best model** as `unet_bestExtended.pth` using early stopping

You’ll see progress and loss updates directly in the terminal.

---

4. For Kvasir Dataset
```bash
python trainCohortMul.py
```

The script will:
- Split the dataset into **train/val/test** sets (15% test, 15% val)
- Train the UNet model
- Save the **best model** as `unet_bestKvasir.pth` using early stopping

You’ll see progress and loss updates directly in the terminal.

---

## 📈 Configuration

Edit `utils/config.py` to adjust hyperparameters and paths:

```python
EPOCHS = 120
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
PATIENCE = 20
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"
MODEL_PATH = "unet_best.pth"
```

---

## 🧪 Testing / Inference (Optional)

For Testing the scores on different dataset


1. For Cohort1 Multiplexed
```bash
python testCohortMul.py 
```

It will create the results as txt file in results folder as resultCohortMul.txt. 

2. For Cohort1 NeuN
```bash
python testCohortNeuN.py 
```

It will create the results as txt file in results folder as resultCohortNeuN.txt. 

3. For Extended
```bash
python testExtended.py 
```

It will create the results as txt file in results folder as resultExtended.txt. 

4. For Kvasir
```bash
python testCohortNeuN.py 
```

It will create the results as txt file in results folder as resultKvasir.txt. 


---

## 🧱 Model Overview

The architecture is a **standard UNet**:

- Encoder: 4 downsampling stages (64 → 128 → 256 → 512)
- Bottleneck: 1024 filters
- Decoder: skip connections and upsampling
- Output: `n_classes` channels (2 for binary segmentation)

**Loss Function:** Dice Loss  
**Optimizer:** Adam  
**Scheduler:** None (constant LR, but easy to add)

---

## 🧩 Example Training Log

```
Epoch 5/50
Train Loss: 0.3241, Val Loss: 0.2987
✅ Model saved (Epoch 5, Val Loss 0.2987)

Epoch 6/50
Train Loss: 0.3178, Val Loss: 0.2975
✅ Model saved (Epoch 6, Val Loss 0.2975)
```

When early stopping triggers:

```
⛔ Early stopping. Best epoch: 6, Val Loss: 0.2975
```

---

## 💾 Checkpoints

After training, you’ll find:

```
unet_best.pth  # best-performing model weights
```

You can load it later for inference:

```python
model.load_state_dict(torch.load("unet_best.pth"))
```

---

## 🧠 Author Notes

Md Hasibul Islam
MSc. Automotive Software Engineering, TU Chemnitz

- Code and Dataset is used for research work
- Original Dataset Collected from [Mehedi Azim!](https://github.com/MehediAzim)

---

## 🧩 Future Improvements


- Add learning rate scheduler
- Add TensorBoard logging

---

## 🪪 License

MIT License — free to use and modify with credit.

---

