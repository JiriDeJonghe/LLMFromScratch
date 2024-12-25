#ifndef TYPES_H
#define TYPES_H

typedef struct {
    float* weights;
    float bias;
    float output;
    int num_inputs;
    float (*activation)(float);
    float (*derivate_activation)(float);
} Neuron;

typedef struct {
    Neuron* neurons;
    int num_neurons;
} Layer;

typedef struct {
    Layer* layers;
    int num_layers;
} NeuralNetwork;

typedef struct {
    float* weight_gradients; // Gradients for each weight
    float error; // Error for this node; important for backprop
} NeuronGradients;

typedef struct {
    NeuronGradients* neuron_gradients;
    int num_neurons;
} LayerGradients;

typedef struct {
    LayerGradients* layer_gradients;
    int num_layers; 
} NetworkGradients;

#endif
