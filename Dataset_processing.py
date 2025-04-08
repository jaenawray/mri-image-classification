import os
import numpy as np
import pandas as pd
from tqdm import tqdm  # for progress bars during loading


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


from PIL import Image


# Image Handling
from PIL import Image
import cv2

# Plotting and visualization
import matplotlib.pyplot as plt


# Paths
image_folder = 'Labeled MRI Brain Tumor Dataset.v1-version-1.tensorflow/train'
csv_file1_train = 'Labeled MRI Brain Tumor Dataset.v1-version-1.tensorflow/train/_annotations.csv'
img_size = (640, 640)

# Load the CSV with correct headers
df_train = pd.read_csv(csv_file1_train)
df_train.columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']


# Drop duplicate filenames (since object detection can have multiple entries per image)
df_unique = df_train.drop_duplicates(subset=['filename'])

# Encode labels
label_encoder = LabelEncoder()
df_unique['label_encoded'] = label_encoder.fit_transform(df_unique['class'])

# Prepare lists
X = []
y = []

# Process images
for idx, row in tqdm(df_unique.iterrows(), total=len(df_unique)):
    img_path = os.path.join(image_folder, row['filename'])
    try:
        img = Image.open(img_path).convert('RGB').resize(img_size)
        img_array = np.array(img) / 255.0  # Normalize
        X.append(img_array)
        y.append(row['label_encoded'])
    except Exception as e:
        print(f"Error loading {img_path}: {e}")