#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "activations.h"
#include "network.h"
#include "types.h"

/**
 * @brief Prints the network weights for all the layers and neurons
 *
 * @param network Pointer to the network of which to print the weights
 */
void print_network_weights(NeuralNetwork *network) {
  for (size_t l = 1; l < network->num_layers; l++) {
    Layer *layer = network->layers[l];
    printf("\nLayer %zu\n", l);
    for (size_t n = 0; n < layer->num_neurons; n++) {
      Neuron *neuron = layer->neurons[n];
      printf("Neuron %zu\n", n);
      for (size_t w = 0; w < neuron->num_inputs; w++) {
        printf("Weight %zu: %f\n", w, neuron->weights[w]);
      }
      printf("Bias: %f\n", neuron->bias);
    }
  }
}

// Network Creation

/**
 * @brief Creates and allocates memory for a Neuron
 *
 * @param num_inputs The number of incoming edges this neuron has
 * @param initial_values the initial values of the weights of the neuron. The
 * length of the array should be equal to the number of inputs
 * @param is_frozen True if the weights of this neuron are frozen and should not
 * be changed during training, otherwise False
 *
 * @return Neuron* Pointer to the created Neuron, NULL if allocation fails
 */
Neuron *create_neuron(size_t num_inputs, double *initial_values,
                      bool is_frozen) {
  if (num_inputs <= 0) {
    fprintf(stderr, "Invalid parameters for neuron creation.\n");
    return NULL;
  }

  Neuron *neuron = (Neuron *)malloc(sizeof(Neuron));
  if (neuron == NULL) {
    perror("Memory allocation failed for neuron creation.");
    return NULL;
  }

  neuron->weights = (float *)malloc(num_inputs * sizeof(float));
  if (neuron->weights == NULL) {
    perror("Memory allocation failed for weights.");
    free(neuron);
    return NULL;
  }

  neuron->num_inputs = num_inputs;
  neuron->output = 0.0f;
  neuron->frozen = is_frozen;

  for (size_t i = 0; i < num_inputs; i++) {
    neuron->weights[i] =
        initial_values ? initial_values[i] : (float)random() / RAND_MAX - 0.5f;
  }
  neuron->bias = initial_values
                     ? initial_values[num_inputs]
                     : (float)random() / RAND_MAX - 0.5f; // Initial bias value

  return neuron;
}

/**
 * @brief Creates and allocates memory for a Layer
 *
 * @param num_neurons The number of neurons there are in this layer
 * @param num_inputs_per_neuron The number of inputs the neurons in this layer
 * should have. We assume a fully connected NN so the number is the same for
 * all neurons in this layer and should be equal to the number of neurons in
 * the previous layer.
 * @param activation The activation function for the neurons in this layer
 * @param activation_derivate The derivate of the activation function for the
 * neurons in this layer
 * @param initial_values Array of initial weights for layers (NULL for random
 * initialization). The length of the array should be equal to the num_neurons.
 * Each element in this array should be equal to num_inputs_per_neuron
 * @param frozen_neurons Array indicating which neurons should be frozen (NULL
 * for none). Should be length of num_neurons.
 *
 * @return Layer* Pointer to the created Layer, NULL if allocation fails
 */
Layer *create_layer(size_t num_neurons, size_t num_inputs_per_neuron,
                    ActivationFunc activation,
                    ActivationFunc activation_derivative,
                    double **initial_values, bool *frozen_neurons) {
  if (num_neurons <= 0 || num_inputs_per_neuron <= 0 || activation == NULL ||
      activation_derivative == NULL) {
    fprintf(stderr, "Invalid parameters for layer creation.\n");
    return NULL;
  }

  Layer *layer = (Layer *)malloc(sizeof(Layer));
  if (layer == NULL) {
    perror("Memory allocation failed for layer.");
    return NULL;
  }

  layer->neurons = (Neuron **)malloc(num_neurons * sizeof(Neuron *));
  if (layer->neurons == NULL) {
    perror("Memory allocation failed for neurons array.");
    free(layer);
    return NULL;
  }

  if (initial_values) {
    for (size_t i = 0; i < num_neurons; i++) {
      if (!initial_values[i]) {
        fprintf(stderr, "Invalid initial values provided for neuron %zu\n", i);
        return NULL;
      }
    }
  }

  layer->activation = activation;
  layer->activation_derivative = activation_derivative;

  layer->num_neurons = num_neurons;
  for (size_t i = 0; i < num_neurons; i++) {
    bool is_frozen = frozen_neurons ? frozen_neurons[i] : false;
    layer->neurons[i] =
        create_neuron(num_inputs_per_neuron,
                      initial_values ? initial_values[i] : NULL, is_frozen);
    if (layer->neurons[i] == NULL) {
      for (size_t j = 0; j < i; j++) {
        free_neuron(layer->neurons[j]);
      }
      free(layer->neurons);
      free(layer);
      return NULL;
    }
  }

  return layer;
}

/**
 * @brief Creates and allocates memory for a Neural Network
 *
 * @param layer_sizes The sizes of the layers (number of neurons in each
 * layer)
 * @param num_layers The number of layers in this Neural Network
 * @param activations The activation functions for the layers in this Neural
 * Network. Last layer doesn't take activation, instead it uses softmax.
 * @param activation_derivates The derivates of the activation functions for
 * the layers in this Neural Network
 * @param initial_values Array of initial weights for layers (NULL for random
 * init). The array should be of length num_layers-1. The element at position i
 * in the array is a matrix of layer_sizes[i] vectors of length
 * layer-sizes[i-1]. Pass NULL for the entire parameter to use random
 * initialization Pass NULL for specific indices to use random initialization
 * for those layers
 * @param frozen_neurons Array of boolean arrays indicating which neurons should
 * be frozen (NULL for none).
 *
 * @return Layer* Pointer to the created Layer, NULL if allocation fails
 */
NeuralNetwork *create_neural_network(size_t *layer_sizes, size_t num_layers,
                                     ActivationFunc *activations,
                                     ActivationFunc *activation_derivatives,
                                     double ***initial_values,
                                     bool **frozen_neurons) {
  if (layer_sizes == NULL || num_layers <= 0 || activations == NULL ||
      activation_derivatives == NULL) {
    fprintf(stderr, "Invalid parameters for network creation.\n");
    return NULL;
  }

  NeuralNetwork *neural_network =
      (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
  if (neural_network == NULL) {
    perror("Memory allocation failed for neural network.");
    return NULL;
  }

  neural_network->layers = (Layer **)malloc(num_layers * sizeof(Layer *));
  if (neural_network->layers == NULL) {
    perror("Memory allocation failed for layer array.");
    free(neural_network);
    return NULL;
  }

  // Creation of input layer
  neural_network->num_layers = num_layers;
  neural_network->layers[0] = create_layer(
      layer_sizes[0], layer_sizes[0], identity, identity_derivate, NULL, NULL);

  if (neural_network->layers[0] == NULL) {
    free(neural_network->layers);
    free(neural_network);
    return NULL;
  }

  // Creation of hidden layers
  for (size_t i = 1; i < num_layers - 1; i++) {
    neural_network->layers[i] =
        create_layer(layer_sizes[i], layer_sizes[i - 1], activations[i - 1],
                     activation_derivatives[i - 1],
                     initial_values ? initial_values[i - 1] : NULL,
                     frozen_neurons ? frozen_neurons[i - 1] : NULL);

    if (neural_network->layers[i] == NULL) {
      for (size_t j = 0; j < i; j++) {
        free_layer(neural_network->layers[j]);
      }
      free(neural_network->layers);
      free(neural_network);
      return NULL;
    }
  }

  // Creation of output layer
  neural_network->layers[num_layers - 1] = create_layer(
      layer_sizes[num_layers - 1], layer_sizes[num_layers - 2], identity,
      identity_derivate, initial_values ? initial_values[num_layers - 2] : NULL,
      frozen_neurons ? frozen_neurons[num_layers - 2] : NULL);

  if (neural_network->layers[num_layers - 1] == NULL) {
    for (size_t j = 0; j < num_layers - 1; j++) {
      free_layer(neural_network->layers[j]);
    }
    free(neural_network->layers);
    free(neural_network);
    return NULL;
  }

  return neural_network;
}

/**
 * @brief Creates and allocates memory for a NeuronGradients
 *
 * @param neuron* Pointer to the neuron to create the gradients of
 *
 * @return NeuronGradients* Pointer to the created NeuronGradients, NULL if
 * allocation fails
 */
NeuronGradients *create_neuron_gradients(Neuron *neuron) {
  if (neuron == NULL) {
    fprintf(stderr, "Invalid parameters for neuron gradients creation.\n");
    return NULL;
  }

  NeuronGradients *neuron_gradients =
      (NeuronGradients *)malloc(sizeof(NeuronGradients));
  if (neuron_gradients == NULL) {
    perror("Memory allocation failed for neuron gradient creation.");
    return NULL;
  }

  neuron_gradients->num_weights = neuron->num_inputs;
  neuron_gradients->error = 0.0f;
  neuron_gradients->weight_gradients =
      (float *)malloc(neuron->num_inputs * sizeof(float));
  if (neuron_gradients->weight_gradients == NULL) {
    perror("Memory allocation failed for weights gradient creation.");
    free(neuron_gradients);
    return NULL;
  }

  return neuron_gradients;
}

/**
 * @brief Creates and allocates memory for a LayerGradients
 *
 * @param layer* Pointer to the layer to create the gradients of
 *
 * @return LayerGradients* Pointer to the created LayerGradients, NULL if
 * allocation fails
 */
LayerGradients *create_layer_gradients(Layer *layer) {
  if (layer == NULL) {
    fprintf(stderr, "Invalid parameters for layer gradients creation.\n");
    return NULL;
  }

  LayerGradients *layer_gradients =
      (LayerGradients *)malloc(sizeof(LayerGradients));
  if (layer_gradients == NULL) {
    perror("Memory allocation failed for layer gradient creation.");
    return NULL;
  }

  layer_gradients->num_neurons = layer->num_neurons;
  layer_gradients->neuron_gradients = (NeuronGradients **)malloc(
      layer->num_neurons * sizeof(NeuronGradients *));
  for (size_t i = 0; i < layer->num_neurons; i++) {
    layer_gradients->neuron_gradients[i] =
        create_neuron_gradients(layer->neurons[i]);
  }

  return layer_gradients;
}

/**
 * @brief Creates and allocates memory for a NetworkGradients
 *
 * @param network* Pointer to the neural network to create the gradients of
 *
 * @return NetworkGradients* Pointer to the created NetworkGradients NULL if
 * allocation fails
 */
NetworkGradients *create_network_gradients(NeuralNetwork *network) {
  if (network == NULL) {
    fprintf(stderr, "Invalid parameters for network gradients creation.\n");
    return NULL;
  }

  NetworkGradients *network_gradients =
      (NetworkGradients *)malloc(sizeof(NetworkGradients));
  if (network_gradients == NULL) {
    perror("Memory allocation failed for network gradient creation.");
    return NULL;
  }

  network_gradients->num_layers = network->num_layers;
  network_gradients->layer_gradients =
      (LayerGradients **)malloc(network->num_layers * sizeof(LayerGradients *));
  for (size_t i = 0; i < network->num_layers; i++) {
    Layer *layer = network->layers[i];
    network_gradients->layer_gradients[i] = create_layer_gradients(layer);
  }

  return network_gradients;
}

// Network Cleanup

/**
 * @brief Frees memory allocated for a neuron
 *
 * @param neuron Pointer to neuron to be freed
 */
void free_neuron(Neuron *neuron) {
  if (neuron == NULL)
    return;
  if (neuron->weights != NULL) {
    free(neuron->weights);
  }
  free(neuron);
}

/**
 * @brief Frees memory allocated for a layer
 *
 * @param layer Pointer to layer to be freed
 */
void free_layer(Layer *layer) {
  if (layer == NULL)
    return;
  if (layer->neurons != NULL) {
    for (size_t i = 0; i < layer->num_neurons; i++) {
      free_neuron(layer->neurons[i]);
    }
    free(layer->neurons);
  }
  free(layer);
}

/**
 * @brief Frees memory allocated for a neural network
 *
 * @param neural_network Pointer to neural network to be freed
 */
void free_neural_network(NeuralNetwork *neural_network) {
  if (neural_network == NULL)
    return;
  if (neural_network->layers != NULL) {
    for (size_t i = 0; i < neural_network->num_layers; i++) {
      free_layer(neural_network->layers[i]);
    }
    free(neural_network->layers);
  }
  free(neural_network);
}

/**
 * @brief Frees memory allocated for a NeuronGradients
 *
 * @param neuron_gradients Pointer to NeuronGradients to be freed
 */
void free_neuron_gradients(NeuronGradients *neuron_gradients) {
  if (neuron_gradients == NULL)
    return;
  if (neuron_gradients->weight_gradients != NULL) {
    free(neuron_gradients->weight_gradients);
  }
  free(neuron_gradients);
}

/**
 * @brief Frees memory allocated for a LayerGradients
 *
 * @param layer_gradients Pointer to LayerGradients to be freed
 */
void free_layer_gradients(LayerGradients *layer_gradients) {
  if (layer_gradients == NULL)
    return;
  if (layer_gradients->neuron_gradients != NULL) {
    for (size_t i = 0; i < layer_gradients->num_neurons; i++) {
      free_neuron_gradients(layer_gradients->neuron_gradients[i]);
    }
    free(layer_gradients->neuron_gradients);
  }
  free(layer_gradients);
}

/**
 * @brief Frees memory allocated for a NetworkGradients
 *
 * @param network_gradients Pointer to NetworkGradients to be freed
 */
void free_network_gradients(NetworkGradients *network_gradients) {
  if (network_gradients == NULL)
    return;
  if (network_gradients->layer_gradients != NULL) {
    for (size_t i = 0; i < network_gradients->num_layers; i++) {
      free_layer_gradients(network_gradients->layer_gradients[i]);
    }
    free(network_gradients->layer_gradients);
  }
  free(network_gradients);
}
