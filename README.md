# Object-Detection-with-Federated-Learning

## Background
Give proof of concept (POC) on the application of federated learning on building object detection model during internship at Audi China R&D.
In this projecct, we adopt Faster R-CNN algorithm in building an object detection model under Tensorflow Federated (TFF) framework. Choose VGG16 as base network.  

- Augment VOC2007 and VOC2012 with duplication and random rotation
- Random assign all data to artificial clients (N= 50, 100, 500)
- Perform FedAvg: update on single client -> update on global server

Functions of important scripts in `codes`:
- `simple_parser.py`: Convert train data format
- `augment.py`: Augment train data with horizontal, vertical flip and random rotation
- `config.py`: Pass train settings 

**Learn how TFF works at simple tasks:**

- Federated learning for image classification using MNIST dataset: https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification
- Federated learning for text generation using Shakespeare dataset: https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation

## Required installation
`h5py`
`tensorflow`
`tensorflow-gpu==1.14.0 `
`Keras==2.0.3`
`numpy`
`opencv-python`
`sklearn`

**See important compatibility at https://github.com/tensorflow/federated#compatibility**
