#include <stdlib.h>
#include <stdio.h>

#include "activations.h"
#include "types.h"
#include "network.h"

// Network Creation

Neuron* create_neuron(int num_inputs, ActivationFunc activation, ActivationFunc derivate_activation) {
    if (num_inputs <= 0 || activation == NULL || derivate_activation == NULL) {
        fprintf(stderr, "Invalid parameters for neuron creation.\n");
        return NULL;
    }

    Neuron* neuron = malloc(sizeof(Neuron));
    if (neuron == NULL) {
        fprintf(stderr, "Memory allocation failed for neuron creation.\n");
        return NULL;
    }

    neuron->weights = malloc(num_inputs * sizeof(float));
    if (neuron->weights == NULL) {
        fprintf(stderr, "Memory allocation failed for weights.\n");
        free(neuron);
        return NULL;
    }

    neuron->num_inputs = num_inputs;
    neuron->bias = (float)rand() / RAND_MAX - 0.5f;
    neuron->activation = activation;
    neuron->derivate_activation = derivate_activation;
    neuron->output = 0.0f;

    for (int i = 0; i < num_inputs; i++) {
        neuron->weights[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    return neuron;
}

Layer* create_layer(int num_neurons, int num_inputs_per_neuron,
                    ActivationFunc activation, ActivationFunc derivate_activation) {
    if (num_neurons <= 0 || num_inputs_per_neuron <= 0) {
        fprintf(stderr, "Invalid parameters for layer creation.\n");
        return NULL;
    }

    Layer* layer = malloc(sizeof(Layer));
    if (layer == NULL) {
        fprintf(stderr, "Memory allocaiton failed for layer.\n");
        return NULL;
    }

    layer->neurons = malloc(num_neurons * sizeof(Neuron));
    if (layer->neurons == NULL) {
        fprintf(stderr, "Memory allocaiton failed for neurons array.\n");
        free(layer);
        return NULL;
    }

    layer->num_neurons = num_neurons;
    for (int i = 0; i < num_neurons; i++) {
        layer->neurons[i] = create_neuron(num_inputs_per_neuron, activation, derivate_activation);
        if (layer->neurons[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free_neuron(layer->neurons[j]);
            }
            free(layer->neurons);
            free(layer);
            return NULL;
        }
    }

    return layer;
}

NeuralNetwork* create_neural_network(int* layer_sizes, int num_layers,
                                     ActivationFunc* activations, ActivationFunc* derivate_activations) {
    if (layer_sizes <= 0 || num_layers <= 0) {
        fprintf(stderr, "Invalid parameters for network creation.\n");
        return NULL;
    }

    NeuralNetwork* neural_network = malloc(sizeof(NeuralNetwork));
    if (neural_network == NULL) {
        fprintf(stderr, "Memory allocaiton failed for neural network.\n");
        return NULL;
    }

    neural_network->layers = malloc(num_layers * sizeof(Layer));
    if (neural_network->layers == NULL) {
        fprintf(stderr, "Memory allocaiton failed for layer array.\n");
        free(neural_network);
        return NULL;
    }

    neural_network->num_layers = num_layers;
    for (int i = 0; i < num_layers; i++) {
        neural_network->layers[i] = create_layer(layer_sizes[i], num_layers, activations[i], derivate_activations[i]);
        if (neural_network->layers[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free_layer(neural_network->layers[j]);
            }
            free(neural_network->layers);
            free(neural_network);
            return NULL;
        }
    }


    return neural_network;
}

NeuronGradients create_neuron_gradients(Neuron *neuron) {
    NeuronGradients neuron_gradients;
    neuron_gradients.error = 0.0f;
    neuron_gradients.weight_gradients = malloc(neuron->num_inputs * sizeof(float));

    return neuron_gradients;
}

LayerGradients create_layer_gradients(Layer* layer) {
    LayerGradients layer_gradients;
    layer_gradients.num_neurons = layer->num_neurons;
    layer_gradients.neuron_gradients = malloc(layer->num_neurons * sizeof(NeuronGradients));
    for (int i = 0; i < layer->num_neurons; i++) {
        layer_gradients.neuron_gradients[i] = create_neuron_gradients(&layer->neurons[i]);
    }

    return layer_gradients;
}

NetworkGradients create_network_gradients(NeuralNetwork* network) {
    NetworkGradients network_gradients;
    network_gradients.num_layers = network->num_layers;
    network_gradients.layer_gradients = malloc(network->num_layers * sizeof(LayerGradients));
    for (int i = 0; i < network->num_layers; i++) {
        Layer* layer = &network->layers[i];
        network_gradients.layer_gradients[i] = create_layer_gradients(layer);
    }

    return network_gradients;
}


// Network Cleanup

void free_neuron(Neuron* neuron) { 
    free(neuron->weights); 
}

void free_layer(Layer* layer) {
    for (int i = 0; i < layer->num_neurons; i++) {
        free_neuron(&layer->neurons[i]);
    }
    free(layer->neurons);
}

void free_neural_network(NeuralNetwork* neural_network) {
    for (int i = 0; i < neural_network->num_layers; i++) {
        free_layer(&neural_network->layers[i]);
    }
    free(neural_network->layers);
}

void free_neuron_gradients(NeuronGradients* neuron_gradients) {
    free(neuron_gradients->weight_gradients);
}

void free_layer_gradients(LayerGradients* layer_gradients) {
    for (int i = 0; i  < layer_gradients->num_neurons; i++) {
        free_neuron_gradients(&layer_gradients->neuron_gradients[i]);
    }
    free(layer_gradients->neuron_gradients);
}

void free_network_gradients(NetworkGradients* network_gradients) {
    for (int i = 0; i < network_gradients->num_layers; i++) {
        free_layer_gradients(&network_gradients->layer_gradients[i]);
    }
    free(network_gradients->layer_gradients);
}

