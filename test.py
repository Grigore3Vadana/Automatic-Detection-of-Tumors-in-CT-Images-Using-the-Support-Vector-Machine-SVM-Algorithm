import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = imread(image_path)
    image = rgb2gray(image)
    image = resize(image, (256, 256))
    return image / 255.0

# Function to load images from a directory
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img = load_and_preprocess_image(os.path.join(folder, filename))
            images.append(img.flatten())  # Flatten the image
            labels.append(label)
    return images, labels

# Load images
tumor_images, tumor_labels = load_images_from_folder(r'C:\Users\Grig\PycharmProjects\ML\Data\tumor', 1)
no_tumor_images, no_tumor_labels = load_images_from_folder(r'C:\Users\Grig\PycharmProjects\ML\Data\no_tumor', 0)

# Combine and split the dataset
data = tumor_images + no_tumor_images
labels = tumor_labels + no_tumor_labels

X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train an SVM model
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)

# Validate the model
y_val_pred = svm.predict(X_val_scaled)
print("Validation Results:")
print(classification_report(y_val, y_val_pred))

# Test the model
y_test_pred = svm.predict(X_test_scaled)
print("Test Results:")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()


from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, svm.decision_function(X_test_scaled))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

precision, recall, _ = precision_recall_curve(y_test, svm.decision_function(X_test_scaled))
average_precision = average_precision_score(y_test, svm.decision_function(X_test_scaled))

plt.figure()
plt.step(recall, precision, where='post', label='Average precision score, AP={0:0.2f}'.format(average_precision))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve')
plt.legend(loc="upper right")
plt.show()


# Function to display image with predicted label
def display_image_with_label(image, label):
    plt.imshow(image.reshape(256, 256), cmap='gray')  # Reshape back to 256x256 for display
    plt.title(f'Predicted Label: {"Tumor" if label == 1 else "No Tumor"}')
    plt.axis('off')
    plt.show()


# Test the model and display results
for i in range(len(X_test_scaled)):
    image = X_test_scaled[i]
    true_label = y_test[i]
    predicted_label = svm.predict([image])[0]

    # Reshape the image for inverse_transform and display
    image_reshaped = image.reshape(1, -1)
    original_image = scaler.inverse_transform(image_reshaped).reshape(256, 256)
    display_image_with_label(original_image, predicted_label)

    # Optionally, you can also print the true label for comparison
    print(f"True Label: {'Tumor' if true_label == 1 else 'No Tumor'}")