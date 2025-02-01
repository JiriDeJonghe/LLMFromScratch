#ifndef NETWORK_H
#define NETWORK_H

#include "activations.h"
#include "types.h"

/**
 * @brief Prints the network weights for all the layers and neurons
 *
 * @param network Pointer to the network of which to print the weights
 */
void print_network_weights(NeuralNetwork *network);

/**
 * @brief Creates and allocates memory for a Neuron
 *
 * @param num_inputs The number of incoming edges this neuron has
 *
 * @return Neuron* Pointer to the created Neuron, NULL if allocation fails
 */
Neuron *create_neuron(size_t num_inputs);

/**
 * @brief Creates and allocates memory for a Layer
 *
 * @param num_neurons The number of neurons there are in this layer
 * @param num_inputs_per_neuron The number of inputs the neurons in this layer
 * should have. We assume a fully connected NN so the number is the same for all
 * neurons in this layer and should be equal to the number of neurons in the
 * previous layer.
 * @param activation The activation function for the neurons in this layer
 * @param activation_derivate The derivate of the activation function for the
 * neurons in this layer
 *
 * @return Layer* Pointer to the created Layer, NULL if allocation fails
 */
Layer *create_layer(size_t num_neurons, size_t num_inputs_per_neuron,
                    ActivationFunc activation,
                    ActivationFunc activation_derivative);

/**
 * @brief Creates and allocates memory for a Neural Network
 *
 * @param layer_sizes The sizes of the layers (number of neurons in each layer)
 * @param num_layers The number of layers in this Neural Network
 * @param activations The activation functions for the layers in this Neural
 * Network
 * @param activation_derivates The derivates of the activation functions for the
 * layers in this Neural Network
 *
 * @return Layer* Pointer to the created Layer, NULL if allocation fails
 */
NeuralNetwork *create_neural_network(size_t *layer_sizes, size_t num_layers,
                                     ActivationFunc *activations,
                                     ActivationFunc *activation_derivative);

/**
 * @brief Creates and allocates memory for a NeuronGradients
 *
 * @param neuron* Pointer to the neuron to create the gradients of
 *
 * @return NeuronGradients* Pointer to the created NeuronGradients, NULL if
 * allocation fails
 */
NeuronGradients *create_neuron_gradients(Neuron *neuron);

/**
 * @brief Creates and allocates memory for a LayerGradients
 *
 * @param layer* Pointer to the layer to create the gradients of
 *
 * @return LayerGradients* Pointer to the created LayerGradients, NULL if
 * allocation fails
 */
LayerGradients *create_layer_gradients(Layer *layer);

/**
 * @brief Creates and allocates memory for a NetworkGradients
 *
 * @param network* Pointer to the neural network to create the gradients of
 *
 * @return NetworkGradients* Pointer to the created NetworkGradients NULL if
 * allocation fails
 */
NetworkGradients *create_network_gradients(NeuralNetwork *network);

/**
 * @brief Frees memory allocated for a neuron
 *
 * @param neuron Pointer to neuron to be freed
 */
void free_neuron(Neuron *neuron);

/**
 * @brief Frees memory allocated for a layer
 *
 * @param layer Pointer to layer to be freed
 */
void free_layer(Layer *layer);

/**
 * @brief Frees memory allocated for a neural network
 *
 * @param neural_network Pointer to neural network to be freed
 */
void free_neural_network(NeuralNetwork *neural_network);

/**
 * @brief Frees memory allocated for a NeuronGradients
 *
 * @param neuron_gradients Pointer to NeuronGradients to be freed
 */
void free_neuron_gradients(NeuronGradients *neuron_gradients);

/**
 * @brief Frees memory allocated for a LayerGradients
 *
 * @param layer_gradients Pointer to LayerGradients to be freed
 */
void free_layer_gradients(LayerGradients *layer_gradients);

/**
 * @brief Frees memory allocated for a NetworkGradients
 *
 * @param network_gradients Pointer to NetworkGradients to be freed
 */
void free_network_gradients(NetworkGradients *network_gradients);

#endif
