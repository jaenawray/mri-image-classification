import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# Setup
training_dir = 'image_data/train_images'
valid_dir = 'image_data/valid_images'
testing_dir = 'image_data/test_images'
image_size = (64, 64)  # experiment 32x32 or 128x128
classes = ['gl', 'me', 'pi', 'no']

# Load and flatten images
def load_images(directory):
    images = []
    labels = []
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            path = os.path.join(directory, file)
            img = imread(path)
            img = resize(img, image_size, anti_aliasing=True)
            img = img.mean(axis=2) if img.ndim == 3 else img  # Convert to grayscale if RGB
            images.append(img.flatten())
            labels.append(file.split('_')[0])
    return np.array(images), np.array(labels)

# Load data
X_train, y_train = load_images(training_dir)
X_valid, y_valid = load_images(valid_dir)
X_test, y_test = load_images(testing_dir)

# Encode labels 
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_valid_enc = encoder.transform(y_valid)
y_test_enc = encoder.transform(y_test)

# Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest 
rf = RandomForestClassifier(n_estimators=10, max_depth=10, max_features='sqrt', random_state=42)
rf.fit(X_train_scaled, y_train_enc)

# Evaluate
val_pred = rf.predict(X_valid_scaled)
test_pred = rf.predict(X_test_scaled)

print("Validation Accuracy:", accuracy_score(y_valid_enc, val_pred))
print("Test Accuracy:", accuracy_score(y_test_enc, test_pred))
print("\nClassification Report:\n", classification_report(y_test_enc, test_pred, target_names=encoder.classes_))

# Feature Importance Heatmap
importances = rf.feature_importances_
heatmap = importances.reshape(image_size[0], image_size[1])  # Reshape to 64x64
plt.figure(figsize=(6, 6))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.title('Random Forest Feature Importance Heatmap')
plt.colorbar(label='Importance')
plt.axis('off')
plt.tight_layout()
plt.show()

"""
# Turn off least important pixels and retrain 
least_important_idx = np.argsort(importances)[:1000]  
X_train_masked = X_train_scaled.copy()
X_train_masked[:, least_important_idx] = 0
X_valid_masked = X_valid_scaled.copy()
X_valid_masked[:, least_important_idx] = 0

rf_masked = RandomForestClassifier(n_estimators=100, random_state=42)
rf_masked.fit(X_train_masked, y_train_enc)
val_masked_pred = rf_masked.predict(X_valid_masked)
print("Validation Accuracy (masked features):", accuracy_score(y_valid_enc, val_masked_pred))
"""
