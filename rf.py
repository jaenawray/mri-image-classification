import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split

# CONFIG 
train_dir = 'image_data/train_images'
valid_dir = 'image_data/valid_images'
test_dir = 'image_data/test_images'
image_size = (128, 128)
n_estimators_options = [5, 7, 10, 15]
max_depth_options = [3, 5, 7, 10]
max_features_options = ['sqrt', 'log2']

# Load images 
def load_images(directory):
    images, labels = [], []
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            path = os.path.join(directory, file)
            img = imread(path)
            img = resize(img, image_size, anti_aliasing=True)
            img = img.mean(axis=2) if img.ndim == 3 else img
            images.append(img.flatten())
            labels.append(file.split('_')[0])
    return np.array(images), np.array(labels)

# Load and prepare training data
X_train_raw, y_train_raw = load_images(train_dir)
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train_raw)

# Split into internal train/val for tuning
X_train, X_val, y_train, y_val = train_test_split(X_train_raw, y_train_enc, test_size=0.2, stratify=y_train_enc, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Hyperparameter tuning
best_acc = 0
best_params = None
best_model = None
results = []

# Store training and validation accuracy for each param combo
train_accs = []
val_accs = []
param_labels = []

print("\nFine-tuning Random Forest (tracking overfitting)...\n")
for n, d, f in product(n_estimators_options, max_depth_options, max_features_options):
    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Predict on both training and validation sets
    train_preds = rf.predict(X_train_scaled)
    val_preds = rf.predict(X_val_scaled)
    
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    param_labels.append(f"{n}T | d={d} | f={f}")
    
    print(f"{param_labels[-1]} -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Track best model
    if (val_acc > best_acc) and ((train_acc - val_acc) <= 0.1):
        best_acc = val_acc
        best_params = (n, d, f)
        best_model = rf


# Plotting train vs validation accuracy
x = np.arange(len(param_labels))
width = 0.35

plt.figure(figsize=(14, 6))
overfit_flags = [(tr - va) > 0.1 for tr, va in zip(train_accs, val_accs)]
colors = ['red' if flag else 'green' for flag in overfit_flags]

plt.bar(x - width/2, train_accs, width, label='Train Accuracy', color='gray')
plt.bar(x + width/2, val_accs, width, label='Validation Accuracy', color=colors)


plt.xticks(x, param_labels, rotation=45, ha='right', fontsize=9)
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy Across Hyperparameters')
plt.legend()
plt.tight_layout()
plt.show()


print(f"\nBest Model: n_estimators={best_params[0]}, max_depth={best_params[1]}, max_features={best_params[2]}")
print(f"Validation Accuracy: {best_acc:.4f}")

# Retrain on full training data (train + valid)
X_full = np.vstack([X_train_raw, load_images(valid_dir)[0]])
y_full = np.concatenate([y_train_raw, load_images(valid_dir)[1]])
y_full_enc = encoder.transform(y_full)

X_full_scaled = scaler.fit_transform(X_full)

rf_final = RandomForestClassifier(
    n_estimators=best_params[0],
    max_depth=best_params[1],
    max_features=best_params[2],
    random_state=42
)
rf_final.fit(X_full_scaled, y_full_enc)

# Evaluate on test set
X_test, y_test = load_images(test_dir)
y_test_enc = encoder.transform(y_test)
X_test_scaled = scaler.transform(X_test)
y_pred = rf_final.predict(X_test_scaled)

# Accuracy comparison
train_preds = rf_final.predict(X_full_scaled)
train_acc = accuracy_score(y_full_enc, train_preds)
test_acc = accuracy_score(y_test_enc, y_pred)

print(f"\nTraining Accuracy (on full train+val set): {train_acc:.4f}")
print(f"\nFinal Test Accuracy (best generalizing model): {test_acc:.4f}")

# Check for overfitting
if train_acc - test_acc > 0.01:
    print("Overfitting detected: training accuracy is significantly higher than test accuracy.")
else:
    print("No significant overfitting detected.")

# Print classification report BEFORE optimization
print("\nClassification Report (before feature optimization):\n")
print(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature Importance Heatmap 
importances = rf_final.feature_importances_
heatmap = importances.reshape(image_size[0], image_size[1])
plt.figure(figsize=(6, 6))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.title('Random Forest Feature Importance Heatmap')
plt.colorbar(label='Importance')
plt.axis('off')
plt.tight_layout()
plt.show()

# Mask least important features and evaluate again 
least_important_idx = np.argsort(importances)[:1000]
X_full_masked = X_full_scaled.copy()
X_full_masked[:, least_important_idx] = 0

X_test_masked = X_test_scaled.copy()
X_test_masked[:, least_important_idx] = 0

rf_masked = RandomForestClassifier(n_estimators=best_params[0], max_depth=best_params[1], max_features=best_params[2], random_state=42)
rf_masked.fit(X_full_masked, y_full_enc)
masked_pred = rf_masked.predict(X_test_masked)

print("Test Accuracy (masked features):", accuracy_score(y_test_enc, masked_pred))
print("\nClassification Report (after masking least important features):\n")
print(classification_report(y_test_enc, masked_pred, target_names=encoder.classes_))
