import pandas as pd
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

training_dir = 'image_data/train_images'
valid_dir = 'image_data/valid_images'
testing_dir = 'image_data/test_images'
image_size = (64, 64)
classes = ['gl', 'me', 'pi', 'no']

def load_images(directory):
    images = []
    labels = []

    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            path = os.path.join(directory, file)
            image = imread(path)

            # resize to get fixed size inputs for the model, flatten to get a 1D feature vector (each pixel is a feature)
            image = resize(image, image_size).flatten()

            label = file.split('_')[0]

            images.append(image)
            labels.append(label)
        
    return np.array(images), np.array(labels)

X_train, y_train = load_images(training_dir)
X_test, y_test = load_images(testing_dir)
X_valid, y_valid = load_images(valid_dir)

# encode the label sets: 'gl' -> 0, 'me' -> 1, etc
encoder = LabelEncoder()
y_train_encoder = encoder.fit_transform(y_train) # let the encoder discover the label numbers in the set 
y_test_encoder = encoder.transform(y_test)
y_valid_encoder = encoder.transform(y_valid)

# instantiate a scaler to even out pixel value distribution, ensuring that certain features don't dominate learning 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit the scaler to the training data, then transform it
X_test_scaled = scaler.transform(X_test)
X_valid_scaled = scaler.transform(X_valid)

# create the support vector classifier, and fit it using the scaled images and encoded labels
# default values: C = 1.0, kernel = rbf, gamma = scale
svm = SVC()
svm.fit(X_train_scaled, y_train_encoder)

y_valid_pred = svm.predict(X_valid_scaled)
print('Validation Accuracy:', accuracy_score(y_valid_encoder, y_valid_pred))

y_test_pred = svm.predict(X_test_scaled)
print('Testing Accuracy:', accuracy_score(y_test_encoder, y_test_pred))

print(classification_report(y_test_encoder, y_test_pred, target_names=encoder.classes_))