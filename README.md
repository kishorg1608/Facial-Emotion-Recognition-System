
# Facial Emotion Recognition System

## Overview

This Facial Emotion Recognition System is designed to identify and recognize emotions from facial expressions in real-time. It utilizes a Convolutional Neural Network (CNN) implemented using the Keras library with TensorFlow as the backend. The model is trained on the FER-2013 dataset, consisting of grayscale images representing seven different emotions.

## Installation

Ensure you have Python installed on your machine, and then install the required packages:

```bash
pip install tensorflow opencv-python
```

## Training

1. **Data Preprocessing:**
   - The training and testing datasets are loaded and preprocessed using Keras' ImageDataGenerator.
   - Images are resized to 48x48 pixels and converted to grayscale.

2. **Model Architecture:**
   - The CNN model includes convolutional layers, max-pooling layers, dense layers, and dropout layers to prevent overfitting.
   - The model is compiled using Stochastic Gradient Descent (SGD) as the optimizer and binary crossentropy as the loss function.

3. **Training the Model:**
   - The model is trained using the `fit_generator` function with a specified number of epochs.
   - Model structure is saved in a JSON file, and weights are saved in an H5 file.

```python
# Example
python train_model.py
```

## Real-Time Emotion Detection

1. **Dependencies:**
   - Ensure that the required packages are installed as mentioned in the installation section.

2. **Run the Emotion Detection System:**
   - Execute the provided Python script for real-time emotion detection.

```python
# Example
python emotion_detection.py
```

3. **Usage:**
   - The system uses the trained model to detect faces in a video feed.
   - Emotion predictions are displayed in real-time with bounding boxes around detected faces.

## Acknowledgments

- [FER-2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data): Kaggle dataset used for training.

Feel free to customize the system and experiment with different models to improve emotion recognition accuracy.
```

Note: Ensure that you provide the correct paths to your training and testing datasets in the code before running the training script.
