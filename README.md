# Brain Tumor Detection using SVM Algorithm

## Overview

This Python script uses Support Vector Machines (SVM) for the classification of medical images into two categories: with tumor (tumor) and without tumor (no_tumor). The script includes functions for image loading, preprocessing, feature scaling, model training, validation, testing, and visualization of results. You can find more detailed project information in the attached PDF 'Detectarea Automata a Tumorilor_Vadana Ioan-Grigore'.

## Requirements

Python 3.x
Libraries: numpy, scikit-image, scikit-learn, matplotlib, seaborn

## Features

Image Preprocessing: Converts images to grayscale, resizes them to 256x256, and normalizes pixel values.
Data Preparation: Loads images from specified directories, labels them, and splits them into training, validation, and test sets.
Data Standardization: Standardizes the data using StandardScaler from scikit-learn.
SVM Model: Trains an SVM classifier with a linear kernel.
Model Evaluation: Evaluates the model using accuracy, precision, recall, confusion matrix, ROC curve, and precision-recall curve.
Visualization: Visualizes confusion matrix, ROC curve, and precision-recall curve using matplotlib and seaborn.
Image Display with Predictions: Displays images with predicted labels.

## Usage

Place your dataset in two separate folders, one for tumor and another for no_tumor images.
Update the folder paths in the script:

tumor_images, tumor_labels = load_images_from_folder('path_to_tumor_images', 1)
no_tumor_images, no_tumor_labels = load_images_from_folder('path_to_no_tumor_images', 0)

Run the script to train and test the SVM model on your data.

## Functions

load_and_preprocess_image(image_path): Loads and preprocesses a single image.
load_images_from_folder(folder, label): Loads all images from a given folder with a specified label.
display_image_with_label(image, label): Displays an image with its predicted label.

## Notes
The script assumes images are in JPEG format. Modify the load_images_from_folder function if your dataset is in a different format.
The paths in load_images_from_folder need to be updated to match your dataset's location.

## Disclaimer
The performance of the model depends on the quality and size of the dataset. This script is intended for educational purposes and may require adjustments for production use.

# License

- This project is protected by copyright and is not available under any public license. All rights are reserved. No part of this project may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without prior written permission from the author.

- © 2024 Vadana Ioan-Grigore. All rights reserved.

# Contact
For support or to report issues, please email grigorevadana3@gmail.com


