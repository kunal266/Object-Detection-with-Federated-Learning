# Object-Detection-with-Federated-Learning

## Background
Give proof of concept (POC) on the application of federated learning on autonomous driving during internship at Audi China R&D.
In this projecct, we adpoted Faster R-CNN from the object detection algorithms into Tensorflow Federated (TFF) framework.
- To simulate real-world scenario, VOC2007 and VOC2012 were augmented with duplication and random rotation.
- Data and object distribution 
- Simulated client and server

The repository contains:
- Image classification and text generation examples in federated learning
- Runing environment for tradition faster r-cnn
- Runing environment for federated faster r-cnn
- Augmented VOC2007 and VOC2012 data 

Functions of scripts in `codes`:
- `simple_parser.py`:
- `augment.py`:

**Learn how TFF works at simple tasks:**

- Federated learning in image classification using MNIST dataset: https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification
- Federated learing in text generation using Shakespeare dataset: https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation

## Required installation
`h5py`
`tensorflow`
`tensorflow-gpu==1.14.0 `
`Keras==2.0.3`
`numpy`
`opencv-python`
`sklearn`

**See important compatiility at https://github.com/tensorflow/federated#compatibility**
