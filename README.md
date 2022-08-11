# Neural_Network

## Table of contents
* [General infos](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Remarks](#Remarks)

## General Info
This program is a simple framework to create and train neural networks in C/C++. It was inspired by the Keras sequential model framework with dense layers for Python. Inside the main.cpp program you will find examples of how to use the framework to create sequential models and to train them for classification and regression problems. The theory was based on the content provided by Andrew Ng. course on Coursera, "Neural Networks and Deep Learning". The framework can train neural networks through the gradient descent optimizer, where the gradients are calculated through forward and subsequent backward propagation algorithms. 

 ## Technologies
 This program was made using C/C++.
 
 ## Setup
It is possible to implement and train regression and classification models using neural networks. A class called "matrix" was made intended to simulate the NumPy library functionalities in Python. The dataset must be converted into matrix objects to feed the networks. Following the notation though by Andrew Ng., the input dataset matrix needs to have the dimensions (number of features, number of samples), e.g., each feature of your model must be in the lines of this matrix and the samples are spread over the columns. The same for the output dataset, (number of outputs, number of samples). To create the sequential model, you can instantiate an object from the network class, passing as an argument the number of features of your input dataset, then use the function "add_Dense" passing as arguments the number of neurons you want, the activation function (sigmoid, tanh, softmax, relu or None = Linear). After it, you use the function "train" passing as arguments the input and output datasets, the number of epochs, the cost function ("mse" for regression, and "cross-entropy" for classification) and the batch size to use and also if you want to shuffle the data. To test your model you can use the function "predict" that will use the last weights found in the training. 
 
 ## Remarks
 It is an experimental program to consolidate the knowledge of basic neural networks theory. Your data should be already normalized and pre-processed before using it. The program needs to be improved yet, with new optimization algorithms, optimizations of the classes, regularization methods, and pre and post-processing functions. 
