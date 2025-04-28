Convolutional Neural Network for Ultrasound Liver Image Classification

Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify grayscale ultrasound liver images from the Annotated Ultrasound Liver Images Dataset on Kaggle. The model categorizes images into one of three classes, likely representing different liver conditions (e.g., normal, abnormal, or specific pathologies), leveraging a deep learning architecture optimized for medical image classification.

Dataset





Source: The dataset is sourced from Kaggle https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset?resource=download and stored locally at "D:/New folder (3)/7272660".



Description: Contains annotated grayscale ultrasound images of the liver, organized into three classes based on directory structure.



Image Specifications: Images are grayscale, resized to 256x256 pixels.



Data Split: Split into 80% training and 20% validation sets using a validation split with a fixed seed (123) for reproducibility.



Labeling: Labels are inferred from the directory structure with categorical label mode.



Batching: Processed in batches of 32 images for efficient training.



Model Architecture

The CNN model is built using Keras' Sequential API with the following layers:





Conv2D: 32 filters (3x3) with ReLU activation, accepting grayscale input (256x256x1).



MaxPooling2D: 2x2 pooling to reduce spatial dimensions.



Conv2D: 64 filters (3x3) with ReLU activation.



MaxPooling2D: 2x2 pooling for further dimensionality reduction.



Flatten: Converts 2D feature maps into a 1D vector.



Dense: 128 units with ReLU activation for high-level feature processing.



Dropout: 50% dropout rate to prevent overfitting.



Dense: Output layer with 3 units and softmax activation for multi-class classification.

Training





Optimizer: Adam optimizer for efficient gradient-based optimization.



Loss Function: Categorical cross-entropy, appropriate for multi-class classification with softmax output and categorical labels.



Metrics: Accuracy is tracked during training and validation.



Epochs: Trained for 10 epochs.



Validation: Validation dataset monitors performance to prevent overfitting.

Evaluation





Evaluated on the validation dataset to compute final loss and accuracy.



Results are printed as "Validation Loss" and "Accuracy" to assess model performance.

Model Saving





Trained model is saved to "d:/my model.keras" for future inference or deployment.

