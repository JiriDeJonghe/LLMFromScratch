#ifndef TYPES_H
#define TYPES_H

#include <stddef.h>
#include <stdlib.h>

typedef struct Neuron {
    float* weights;
    float bias;
    float output;
    size_t num_inputs;
} Neuron;

typedef struct Layer {
    Neuron** neurons;
    size_t num_neurons;
    float (*activation)(float);
    float (*activation_derivative)(float);
} Layer;

typedef struct NeuralNetwork {
    Layer** layers;
    size_t num_layers;
} NeuralNetwork;

typedef struct {
    float* weight_gradients; // Gradients for each weight
    float error; // Error for this node; important for backprop
    size_t num_weights;
} NeuronGradients;

typedef struct {
    NeuronGradients** neuron_gradients;
    size_t num_neurons;
} LayerGradients;

typedef struct {
    LayerGradients** layer_gradients;
    size_t num_layers; 
} NetworkGradients;

#endif
