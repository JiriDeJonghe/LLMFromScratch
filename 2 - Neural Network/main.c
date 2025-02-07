#include <stdio.h>
#include <stdlib.h>

#include "activations.h"
#include "dataset.h"
#include "network.h"
#include "propagation.h"
#include "types.h"

enum OutputType { PROBS, BINARY };

float calculate_accuracy(NeuralNetwork *network, Dataset *dataset,
                         size_t num_cases,
                         LayerActivationFunc *layer_activation_func) {
  int correct = 0;

  for (size_t i = 0; i < num_cases; i++) {
    forward_propagate(network, dataset->samples[i].inputs,
                      layer_activation_func);

    Layer *output_layer = network->layers[network->num_layers - 1];
    float prediction = output_layer->neurons[0]->output;
    float expected = dataset->samples[i].expected_outputs[0];

    if ((prediction >= 0.5 && expected == 1) ||
        (prediction < 0.5 && expected == 0)) {
      correct++;
    }
  }

  return (float)correct / num_cases;
}

int main() {
  // Change these two values to what you want to use. IMPORTANT: make sure that
  // the dataset corresponds with the output_type, e.g., if training a
  // classifier that has one output neuron, use binary.
  enum OutputType output_type = BINARY;
  char *dataset_name = "fraud_long";

  // You can also change the following for other execution parameters
  const size_t epochs = 1000;
  const float learning_rate = 0.0001f;

  LayerActivationFunc *layer_activation_func;
  if (output_type == PROBS) {
    layer_activation_func = softmax;
  } else if (output_type == BINARY) {
    layer_activation_func = heaviside_step_function;
  }
  size_t layer_sizes[] = {2, 2, 1};      // Input, (Hidden)*, Output
  ActivationFunc activations[] = {relu}; // Only for the hidden layers
  ActivationFunc activation_derivatives[] = {
      relu_derivative}; // Only for the hidden layers

  NeuralNetwork *network = create_neural_network(layer_sizes, 3, activations,
                                                 activation_derivatives);

  // ---- Below no touchy required ----
  if (network == NULL) {
    fprintf(stderr, "Failed to create neural network\n");
    return 1;
  }

  NetworkGradients *gradients = create_network_gradients(network);
  if (gradients == NULL) {
    fprintf(stderr, "Failed to create gradients\n");
    free_neural_network(network);
    return 1;
  }

  Dataset *dataset = create_example_dataset(dataset_name);
  const size_t num_test_cases = dataset->num_samples;

  printf("Training started...\n");
  for (size_t epoch = 0; epoch < epochs; epoch++) {
    float total_error = 0.0f;

    for (size_t i = 0; i < num_test_cases; i++) {
      forward_propagate(network, dataset->samples[i].inputs,
                        layer_activation_func);
      backward_propagate(network, gradients, dataset->samples[i].inputs,
                         dataset->samples[i].expected_outputs, learning_rate);

      Layer *output_layer = network->layers[network->num_layers - 1];

      for (size_t nb_output_neuron = 0;
           nb_output_neuron < output_layer->num_neurons; nb_output_neuron++) {
        float prediction = output_layer->neurons[nb_output_neuron]->output;
        float expected = dataset->samples[i].expected_outputs[nb_output_neuron];
        total_error += (prediction - expected) * (prediction - expected);
      }
    }

    if ((epoch + 1) % 10 == 0) {
      printf("Epoch %zu, Error: %.4f\n", epoch + 1,
             total_error / num_test_cases);
    }

    if ((epoch + 1) % 100 == 0) {
      print_network_weights(network);
    }
  }

  printf("\nTraining completed. Final results:\n");
  float accuracy = calculate_accuracy(network, dataset, num_test_cases,
                                      layer_activation_func);
  printf("\nFinal accuracy: %.2f%%\n", accuracy * 100);

  free_network_gradients(gradients);
  free_neural_network(network);

  return 0;
}
