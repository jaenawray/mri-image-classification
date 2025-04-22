import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

# initializations 
training_dir = 'image_data/train_images'
valid_dir = 'image_data/valid_images'
testing_dir = 'image_data/test_images'
image_size = (128, 128)
classes = ['gl', 'me', 'pi', 'no']

# method to load images from a given directory
def load_images(directory):
    images = []
    labels = []

    for file in os.listdir(directory):
        if file.endswith('.jpg'): # ignore any non jpg files
            path = os.path.join(directory, file)
            image = imread(path, as_gray=True) # make greyscale

            # resize to get fixed size inputs for the model, flatten to get a 1D feature vector (each pixel is a feature)
            image = resize(image, image_size).flatten()
            #print(f"{file}: shape {image.shape}")

            label = file.split('_')[0] # according to naming conventions

            images.append(image)
            labels.append(label)
        
    return np.array(images), np.array(labels)

# load the images for the different sets
X_train, y_train = load_images(training_dir)
X_test, y_test = load_images(testing_dir)
X_valid, y_valid = load_images(valid_dir)

# encode the label sets: 'gl' -> 0, 'me' -> 1, etc
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train) # let the encoder discover the label numbers in the set 
y_test_encoded = encoder.transform(y_test)
y_valid_encoded = encoder.transform(y_valid)

# instantiate a scaler to even out pixel value distribution, ensuring that certain features don't dominate learning 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit the scaler to the training data, then transform it
X_test_scaled = scaler.transform(X_test)
X_valid_scaled = scaler.transform(X_valid)

# create the support vector classifier, and fit it using the scaled images and encoded labels
# default values: C = 1.0, kernel = rbf, gamma = scale
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train_encoded)

# get the accuracy based on the validation set
y_valid_pred = svm.predict(X_valid_scaled)
print('Validation Accuracy:', accuracy_score(y_valid_encoded, y_valid_pred))

# hyperparameter tuning
params = {'C': [0.1, 1, 5, 10], 'gamma': ['scale', 0.1, 1]}
svm = SVC(kernel = 'rbf', gamma='scale')
grid = GridSearchCV(
    estimator=svm,
    param_grid=params,
    cv=5, # 5-fold cross validation
    verbose=3, # print values as they're tested
    return_train_score=True
)

grid.fit(X_train_scaled, y_train_encoded)

# determine the best parameters and score based on the grid search
print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

# convert the results of the cross validation to a dataframe
cv_results = pd.DataFrame(grid.cv_results_)

# compare the training score and the testing (validation) score from cross validation
# calculate the gap between the values (if gap > 5%, it's overfitting)
cv_results['mean_train_score'] = cv_results['mean_train_score']
cv_results['mean_test_score'] = cv_results['mean_test_score']
cv_results['gap'] = cv_results['mean_train_score'] - cv_results['mean_test_score']

print("\nTrain vs. Validation Accuracy:")
print(cv_results[['params', 'mean_train_score', 'mean_test_score', 'gap']])

# visualize the time it took to train and test each grid search parameter combo
plt.figure(figsize=(10, 6))
plt.barh(
    y=[f"C={row['C']}, γ={row['gamma']}" for idx, row in cv_results.iterrows()],
    width=cv_results['time'],
    color='skyblue'
)
plt.xlabel('Mean Time (s)')
plt.title('Training + Testing Time for Each Param Combo')
plt.tight_layout()
plt.show()

# visualize the accuracies for the different C (0.1, 1, 10) ad gamma values (scale, 0.1, 1)
# C is plotted on the X axis, gamma is shown through colors 
plt.figure(figsize=(10, 6))
sns.barplot(x='C', y='test_score', hue='gamma', data=cv_results, palette='viridis')
plt.title('SVM Test Accuracy by C and Gamma')
plt.ylabel('Test Accuracy')
plt.ylim(0.3, 1.0)
plt.show()
plt.figure(figsize=(8, 5))

# create the full training set (train + valid sets)
X_full = np.vstack((X_train_scaled, X_valid_scaled))
y_full = np.concatenate((y_train_encoded, y_valid_encoded))

# train on the full training set with the best params
"""svm = SVC(**grid.best_params_)"""
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_full, y_full)

# test the model 
y_test_pred = svm.predict(X_test_scaled)
print('Testing Accuracy:', accuracy_score(y_test_encoded, y_test_pred))

# get the classification report and print it
report = classification_report(y_test_encoded, y_test_pred, target_names=encoder.classes_)
print(report)

# plot the confusion matrix as a heatmap to show the accuracy of predictions
cm = confusion_matrix(y_test_encoded, y_test_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Set)')
plt.show()

# convert labels to binary format for plotting ROC/PR curves
y_test_binary = label_binarize(y_test_encoded, classes=[0, 1, 2, 3])
n_classes = y_test_binary.shape[1] # get the number of classes

# use a One Vs Rest classifier to train one binary classifier per class to handle multiple classes
ovr_svm = OneVsRestClassifier(SVC(C=10, kernel='rbf', probability=True, random_state=42))
ovr_svm.fit(X_full, y_full)

# predict the probability for each class for the test set
y_score = ovr_svm.predict_proba(X_test_scaled)

# initialize a dict for the false positive rate (fpr) and true positive rate (tpr) and area under the curve (AUC)
fpr = dict()
tpr = dict()
roc_auc = dict()

# loop through each class 
for i in range(n_classes):
    # get the true binary labels for class i and the predicted probability scores, compute the fpr/tpr
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_score[:, i])

    # calculate the area under the curve for the class
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ['aqua', 'orange', 'blue', 'green'] 
class_names = encoder.classes_ # me, gl, pi, no

# plot the ROC curve for each class and display the AUC value
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

# plot the line for a random classifier
plt.plot([0, 1], [0, 1], 'k--', lw=2)  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC-AUC (SVM)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# get confidence scores for each class for the SVM model
y_scores = svm.decision_function(X_test_scaled)

plt.figure(figsize=(8, 6))

# loop thru the number of classes
for i in range(len(class_names)):
    # calculate the precision recall curve for the binary indicators and confidence scores
    precision, recall, _ = precision_recall_curve(y_test_binary[:, i], y_scores[:, i])

    # calculate avreage precision to summarize 
    ap_score = average_precision_score(y_test_binary[:, i], y_scores[:, i])
    plt.plot(recall, precision, lw=2, label=f"{class_names[i]} (AP={ap_score:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve per Class")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()