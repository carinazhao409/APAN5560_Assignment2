# Assignment 2 - CNN Image Classifier

This project implements and deploys a Convolutional Neural Network
using PyTorch and FastAPI.

## Part 1: CNN Architecture

The CNN accepts 64x64 RGB images and uses the following architecture:

- Conv2D: 16 filters, 3x3 kernel, stride 1, padding 1
- ReLU
- MaxPool2D: 2x2 kernel, stride 2
- Conv2D: 32 filters, 3x3 kernel, stride 1, padding 1
- ReLU
- MaxPool2D: 2x2 kernel, stride 2
- Flatten
- Fully connected layer with 100 units
- ReLU
- Fully connected output layer with 10 units

## Part 2: CIFAR-10 Classification

The CNN is trained on the CIFAR-10 dataset.

The 10 classes are:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

CIFAR-10 images are resized to 64x64 to match the CNN input
architecture.

The trained model is saved as:

`cifar10_cnn.pth`

## API

The trained classifier is deployed using FastAPI.

Start the API with:

```bash
uvicorn main:app --reload