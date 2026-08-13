# Plant Disease Detection Using Deep Learning

## Overview

This project presents a deep learning-based image classification system for identifying plant diseases from leaf images.

The system uses transfer learning with the InceptionV3 architecture to classify plant leaf images into different disease categories.

## Objective

The main objective is to develop an automated computer vision system that can assist in identifying plant diseases from images and provide a predicted disease class with a confidence score.

## Technologies Used

* Python
* TensorFlow
* Keras
* InceptionV3
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Flask

## Dataset

The dataset contains leaf images belonging to multiple plant categories, including:

* Banana
* Chilli
* Corn
* Onion
* Tomato

The images are processed and prepared for deep learning model training and evaluation.

## Methodology

The system follows these major steps:

1. Image dataset collection
2. Image preprocessing
3. Data augmentation
4. Transfer learning using InceptionV3
5. Model training
6. Model validation
7. Performance evaluation
8. Disease prediction through a web application

## Model Architecture

InceptionV3 is used as the backbone network for extracting meaningful visual features from plant leaf images.

The pretrained feature extractor is combined with classification layers to predict the corresponding plant disease category.

## Application

The trained model can be integrated with a Flask-based web application where a user can upload a plant leaf image and receive the predicted disease category.

## Project Structure

```text
plant-disease-detection/
│
├── README.md
├── train.py
├── app.py
├── requirements.txt
├── model/
└── screenshots/
```

## Future Enhancement

Future development can include:

* Increasing the number of plant disease classes
* Improving model accuracy
* Real-time camera-based disease detection
* Mobile application integration
* Explainable AI for prediction interpretation
* Deployment as a cloud-based AI service

## Skills Demonstrated

**Python • Deep Learning • Computer Vision • Transfer Learning • TensorFlow • Keras • Flask • Image Classification**
