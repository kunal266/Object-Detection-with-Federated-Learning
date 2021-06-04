# Object-Detection-with-Federated-Learning

## Background
The Object Detection with Federated Learning project is for giving proof of concept (POC) on the application of federated learning on intelligent connected vehicle networks during my internship at Audi China R&D .
In this projecct, we adpoted Faster R-CNN from the object detection algorithms into Tensorflow Federated (TFF) framework.
- To simulate real-world scenario, VOC2007 and VOC2012 were augmented with duplication and random rotation.
- Data and object distribution 
- Simulated client and server

The repository contains:
- Examples in federated learning
- Runing environment for tradition faster r-cnn
- Runing environment for federated faster r-cnn
- Augmented VOC2007 and VOC2012 data 

## Related Efforts
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
