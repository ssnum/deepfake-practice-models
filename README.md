# Deepfake Practice Models
This is a collection of my code where I practiced using simple AI Models to facilitate my learning within AI. For my parameters, I used both the brightness values per image and the actual image data itself (RGB pixel values). This is included to show my practice code for learning to use pre-built AI models with my dataset prior to developing a Neural Network. Note that code cannot be run without referencing data preprocessing and analysis from the deepfake-data-visualization repository, and to reference this as learning practice.

## Features
- KNN Practice Model (for brightness and real image data)
- Random Forest Classifier (for brightness and real image data)
- Random Forest Classifier Bootstrapping method (for brightness data)
- Logistic Regression (for brightness and real image data)
- Support Vector Classification (for brightness data)
- pre-processing code to show how I accessed and stored real RGB values per image
- model metrics practice for real image data including accuracy, precision, f1 score, recall, confusion matrix, roc curve, and aea under roc curve

## Requirements to Use
- import all neccesary libraries referenced throughout the code
- NOTE: cannot be run without downloading the kaggle dataset

## Credits
- Data Set from Kaggle: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
