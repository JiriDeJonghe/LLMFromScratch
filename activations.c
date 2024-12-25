#include <math.h>

#include "types.h"

float sigmoid(float value) { 
    return 1.0f / (1.0f + exp(-value)); 
}

float sigmoid_derivative(float value) {
    float sig = sigmoid(value);
    return sig * (1.0 - sig);
}


float relu(float value) { 
    return value > 0 ? value : 0; 
}

float relu_derivate(float value) {
    return value > 0 ? 1 : 0;
}

void softmax(Layer* layer) {
    double sum = 0.0f;

    for (int i = 0; i < layer->num_neurons; i++) {
        Neuron neuron = layer->neurons[i];
        neuron.output = exp(neuron.output);
        sum += neuron.output;
    }

    for (int i = 0; i < layer->num_neurons; i++) {
        Neuron neuron = layer->neurons[i];
        neuron.output /= sum;
    }
}
