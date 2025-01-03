# Creating a Transformer from Scratch

The goal of this repo is to be a guide for you to be able to understand, explain and create your own LLM from scratch. This repo will stepwise built an LLM starting from a Perceptron and gradually expend it's capabilities until we get to LLMs. Each chapter will try to give an intuitive explanation as to why change was required, what additions were made to resolve the issues and how to interpet the results. Furthermore, each chapter will also give a more technical deepdive and provide an implementation in C.

## 1 - Perceptron

The first step in creating an LLM is understanding a [Perceptron](./1 - Perceptron/README.md). Simply put, a perceptron is an algorithm used for binary classification by estimating a linear function that divides the search space into two, where samples falling on opposite sides have opposite labels. After estimating the linear function, the perceptron can be used for inference to classify unseen samples.


