# Contradiction Detection Model

## Overview

This repository contains the implementation of a neural network model for detecting contradictions between pairs of sentences. The model incorporates an attention mechanism to enhance accuracy, and the implementation is based on TensorFlow and Keras.

## Table of Contents

- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing](#testing)

## Dependencies

Ensure you have the following dependencies installed:

- pandas
- numpy
- tensorflow
- keras

Install them using:

```bash
pip install pandas numpy tensorflow keras
```

## Dataset

The model is trained and tested on a dataset 'Contradiction Detection' by athu1105 from Kaggle, provided in CSV format. Please place the training data ('train.csv') and testing data ('test.csv') in the same directory as the code.

## Usage

1. Import necessary libraries and set the random seed for reproducibility.
2. Load the dataset using `pd.read_csv`.
3. Preprocess the data by handling missing values and tokenizing/padding sequences.
4. Define the model architecture, including the attention mechanism.
5. Compile the model with the specified optimizer, loss function, and metrics.
6. Train the model using the training data.
7. Evaluate the model on the testing data.
8. Make predictions on new sentences.

## Model Architecture

The model architecture comprises an embedding layer, LSTM layer for sequence understanding, attention mechanism to weigh relevant parts of sentences, global max pooling for feature extraction, dropout layers for regularization, and dense layers for final predictions.

## Training

Adjust hyperparameters such as epochs and batch size according to your preference. The model is trained using the `fit` method.

## Testing

Test the model on new sentences by providing them to the trained model. The output will indicate the probability of contradiction between the sentences.
