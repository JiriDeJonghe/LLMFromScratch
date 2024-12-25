#include "network.h"
#include <stdlib.h>

#include "activations.h"
#include "types.h"

// Network Creation

Neuron create_neuron(int num_inputs, ActivationFunc activation, ActivationFunc derivate_activation) {
    Neuron neuron;
    neuron.num_inputs = num_inputs;
    neuron.weights = malloc(num_inputs * sizeof(float));
    neuron.bias = (float)rand() / RAND_MAX - 0.5f;
    neuron.activation = activation;
    neuron.derivate_activation = derivate_activation;

    for (int i = 0; i < num_inputs; i++) {
        neuron.weights[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    return neuron;
}

Layer create_layer(int num_neurons, int num_inputs_per_neuron,
                    ActivationFunc activation, ActivationFunc derivate_activation) {
    Layer layer;
    layer.num_neurons = num_neurons;
    layer.neurons = malloc(num_neurons * sizeof(Neuron));
    for (int i = 0; i < num_neurons; i++) {
        layer.neurons[i] = create_neuron(num_inputs_per_neuron, activation, derivate_activation);
    }

    return layer;
}

NeuralNetwork create_neural_network(int* layer_sizes, int num_layers,
                                     ActivationFunc* activations, ActivationFunc* derivate_activations) {
    NeuralNetwork neural_network;
    neural_network.num_layers = num_layers;
    neural_network.layers = malloc(num_layers * sizeof(Layer));
    for (int i = 0; i < num_layers; i++) {
        int num_inputs = (i == 0) ? layer_sizes[i] : layer_sizes[i - 1];
        neural_network.layers[i] = create_layer(layer_sizes[i], num_inputs, activations[i], derivate_activations[i]);
    }

    return neural_network;
}

NeuronGradients create_neuron_gradients(Neuron *neuron) {
    NeuronGradients neuron_gradients;
    neuron_gradients.bias_gradient = 1;
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

void free_network_gradients(NetworkGradients* network_gradients) {
    for (int i = 0; i < network_gradients->num_layers; i++) {
        free_layer_gradients(&network_gradients->layer_gradients[i]);
    }
    free(network_gradients->layer_gradients);
}

