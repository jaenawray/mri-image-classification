import os
from sklearn.model_selection import train_test_split
import shutil


image_dir = '/Users/mayasachidanand/Documents/GitHub/mri-image-classification/images_to_split'
output_dir = '/Users/mayasachidanand/Documents/GitHub/mri-image-classification/image_data'

# Collect image paths and labels
images = []
labels = []

for file in os.listdir(image_dir):
    if file.endswith('.jpg'):
        label = file.split('_')[0]  
        if label in ['me', 'gl', 'pi', 'no']:
            images.append(file)
            labels.append(label)

# First split into train+val and test (90 / 10)
X_temp, X_test, y_temp, y_test = train_test_split(
    images, labels, test_size=0.10, stratify=labels, random_state=42
)

# Then split train+val into train and val (approx. 70/20 overall)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=2/9, stratify=y_temp, random_state=42
)

# Helper to copy files into appropriate folders
def copy_files(file_list, label_list, split):
    for file, label in zip(file_list, label_list):
        src = os.path.join(image_dir, file)
        dst = os.path.join(output_dir, split, file)
        shutil.copy(src, dst)

# Copy all files
copy_files(X_train, y_train, 'train_images')
copy_files(X_val, y_val, 'valid_images')
copy_files(X_test, y_test, 'test_images')

print("Dataset split complete.")