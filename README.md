# Facial Emotion Detection Project

## Overview

This project is a facial emotion detection system that uses deep learning techniques to recognize and classify human emotions based on facial expressions. It is designed to provide a reliable and efficient solution for detecting emotions such as happiness, sadness, anger, and more in real-time or from images.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)

## Features

- Real-time emotion detection from live video streams.
- Emotion classification for individual images.
- Supports multiple emotions, including happiness, sadness, anger, and more.

## Installation

To use this facial emotion detection system, follow these installation steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/tajammulbasheer/facial_emotion_detection.git
   ```

2. Navigate to the project directory:

   ```bash
   cd facial_emotion_detection
   ```

3. Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```


## Usage

You can use the facial emotion detection system as a standalone application or integrate it into your own projects. Here's how to run the system:

1. To perform real-time emotion detection, run the following command:

   ```bash
   python predict_oncam.py --model_path 'path to saved model'
   ```

2. To classify the emotion in an individual image, use the following command:

   ```bash
   python predict_onimage.py --model_path 'path to saved model' --image your_image.jpg
   ```

## Model Training

If you want to train your own emotion detection model, follow these steps:

1. Prepare your dataset of facial expressions with corresponding emotion labels.

2. Modify the model architecture in `train_build_model.py` to suit your requirements.

3. Train your model using your dataset:

   ```bash
   python train_build_model.py --data_path /path/to/your/dataset
   ```
4. After training, save the model weights and update the configuration in the code to use your custom model.
