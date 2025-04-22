# mri-image-classification

# MRI Image Classification

## Brain Tumor Classification from MRI Scans

This repository implements and compares three machine learning models for classifying brain tumors from MRI scans:

- **Support Vector Machine (SVM)**
- **Convolutional Neural Network (CNN)**
- **Random Forest (RF)**

Each model is trained to classify images into one of four classes: **glioma**, **meningioma**, **pituitary**, and **no tumor**.

## Requirements
Python 3.8+

## Required libraries:
numpy
matplotlib
seaborn
scikit-learn
scikit-image
torch
torchvision


## Model Descriptions

### SVM Classifier

- **Script:** `svm.py`
- Preprocessed features from grayscale MRI images
- Grid search over kernel types and regularization parameters
- **Outputs:** classification report and confusion matrix

### CNN Classifier

- **Script:** `cnn.ipynp`
- Architecture: 3 convolutional blocks with batch normalization, dropout, and 2 fully connected layers
- Optimizer: AdamW
- Early stopping based on validation accuracy
- **Outputs:** accuracy/loss plots, confusion matrix, classification report

---
### Random Forest Classifier

- **Script:** `rf.py`
- Grid search over:
  - Number of trees (`n_estimators`)
  - Tree depth (`max_depth`)
  - Max features (`max_features`)
- Tracks overfitting by comparing training and validation accuracy
- Retrains on full train+validation set before testing
- **Outputs:**
- Train vs Validation Accuracy plot (`rf_fig_1.png`)
- Confusion matrix (`rf_figure_2.png`)
- Feature importance heatmap (`rf_fig_3.png`)
- Precision–Recall curves (`rf_precision_recall_curve.png`)
- ROC curves (`rf_roc_curve.png`)
- Post-feature-masking performance evaluation

