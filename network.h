#ifndef NETWORK_H
#define NETWORK_H

#include "types.h"
#include "activations.h"

Neuron create_neuron(int num_inputs, ActivationFunc activation, ActivationFunc derivate_activation);
Layer create_layer(int num_neurons, int num_inputs_per_neuron, ActivationFunc activation, ActivationFunc derivate_activation);
NeuralNetwork create_neural_network(int *layer_sizes, int num_layers, ActivationFunc *activations, ActivationFunc* derivate_activations);

void free_neuron(Neuron* neuron);
void free_layer(Layer* layer);
void free_neural_network(NeuralNetwork* neural_network);

NeuronGradients create_neuron_gradients(Neuron* neuron);
LayerGradients create_layer_gradients(Layer* layer);
NetworkGradients create_network_gradients(NeuralNetwork* network);

void free_neuron_gradients(NeuronGradients* neuron_gradients);
void free_layer_gradients(LayerGradients* layer_gradients);
void free_netork_gradients(NetworkGradients* network_gradients);


#endif
