# Student Name:- Amey Banarase
# Student Id :- bitsom_ba_2511968

# Part 2: Computer Vision Problem Formulation and CNN Prototype

## Project Overview
This project builds a CNN-based image classifier to detect **manufacturing surface defects**. The model classifies product images into 4 categories: normal, scratch, stain, and dent.

## Dataset Source
**Dataset:** Synthetic Manufacturing Defect Image Dataset  
**Source:** BITSoM Module 5 - Part 2 Shared Google Drive Folder  
https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

## Dataset Description
| Property | Details |
|---|---|
| Total Images | 480 |
| Number of Classes | 4 |
| Images per Class | 120 |
| Image Type | RGB |
| Task | Multi-class Image Classification |

### Classes
| Class | Description |
|---|---|
| normal | Product surface without any defect |
| scratch | Surface with scratch-like marks |
| dent | Surface with circular dent-like marks |
| stain | Surface with colored stain-like marks |

## Repository Structure
```
part-2-cnn-computer-vision/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── accuracy_loss_curves.png
    ├── confusion_matrix.png
    ├── class_distribution.png
    └── sample_images.png
```

## Tasks Completed

### Task 1: Problem Identification
- **Problem Type:** Image Classification (Multi-class)
- Each image is assigned one of 4 class labels
- No bounding boxes or pixel masks required
- CNN is ideal for this structured image classification task

### Task 2: Dataset Exploration
- 480 total images across 4 classes
- Perfectly balanced dataset — 120 images per class
- No class imbalance issue
- Visualized sample images from each class

### Task 3: Image Preprocessing
- Resized all images to **64x64 pixels**
- Normalized pixel values to **[0, 1]** range
- Applied **one-hot encoding** for class labels
- **Train-Test Split:** 80% training (384), 20% testing (96)
- Used `stratify=y` to maintain class balance in splits

### Task 4: CNN Model Architecture
```
Input: (64, 64, 3)
↓
Conv2D(32, 3x3) + ReLU + MaxPooling(2x2)
↓
Conv2D(64, 3x3) + ReLU + MaxPooling(2x2)
↓
Conv2D(128, 3x3) + ReLU + MaxPooling(2x2)
↓
Flatten
↓
Dense(128) + ReLU + Dropout(0.4)
↓
Output: Dense(4) + Softmax
```
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Output Activation:** Softmax (multi-class)

### Task 5: Model Training and Evaluation
- Trained for 30 epochs with batch size 32
- Evaluated using accuracy, loss curves, confusion matrix, and sample predictions

### Task 6: CNN Concept Explanation

**What is Convolution?**  
A filter (e.g., 3x3) slides over the image performing element-wise multiplication to detect patterns like edges and textures, producing a feature map.

**Why is Pooling Used?**  
MaxPooling reduces feature map size, lowering computation and making the model robust to small shifts or rotations in the image.

**Why is ReLU Used?**  
ReLU introduces non-linearity, is computationally cheap, and avoids the vanishing gradient problem that affects sigmoid/tanh in deep networks.

**Why CNNs over Feed-Forward Networks for Images?**  
CNNs use parameter sharing and spatial awareness — they understand that nearby pixels are related, making them far more efficient and accurate for image data.

### Task 7: Business Use Case — Manufacturing

**Problem:** Manual visual inspection of products on assembly line is slow and error-prone.

**AI Solution:** CNN camera system classifies each product in real-time as normal or defective (scratch/dent/stain), automatically routing defective items for rework.

**Benefits:**
- Faster inspection (thousands per hour)
- Consistent accuracy (no human fatigue)
- Reduced cost and improved product quality

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Technologies Used
- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
