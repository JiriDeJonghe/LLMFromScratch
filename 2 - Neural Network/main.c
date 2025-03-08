#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    } else {
      /*printf("\n\nInput %f, %f\n", dataset->samples[i].inputs[0],*/
      /*       dataset->samples[i].inputs[1]);*/
      /*printf("Output %f\n", *dataset->samples[i].expected_outputs);*/
      /*for (size_t l = 1; l < network->num_layers; l++) {*/
      /*  Layer *layer = network->layers[l];*/
      /*  printf("\nLayer %zu\n", l);*/
      /*  for (size_t n = 0; n < layer->num_neurons; n++) {*/
      /*    Neuron *neuron = layer->neurons[n];*/
      /*    printf("Neuron %zu\n", n);*/
      /*    for (size_t w = 0; w < neuron->num_inputs; w++) {*/
      /*      printf("Weight %zu: %f\n", w, neuron->weights[w]);*/
      /*    }*/
      /*    printf("Bias: %f\n", neuron->bias);*/
      /*    printf("Output: %f\n", neuron->output);*/
      /*  }*/
      /*}*/
    }
  }

  return (float)correct / num_cases;
}

int main(int argc, char *argv[]) {
  char *dataset_name = "fraud";
  enum OutputType output_type = BINARY;

  if (argc > 1 &&
      (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
    printf("Usage: %s [dataset] [output_type] [heaviside_example]\n", argv[0]);
    printf("  dataset: Name of the dataset (default: fraud)\n");
    printf("  output_type: binary or probs (default: binary)\n");
    printf("  heaviside_example: whether to use the heaviside example\n");
    return 0;
  }

  if (argc > 1) {
    if (strcmp(argv[1], "fraud") == 0) {
      dataset_name = "fraud";
    } else if (strcmp(argv[1], "fraud_long") == 0) {
      dataset_name = "fraud_long";
    } else {
      fprintf(stderr, "Invalid dataset name. Use 'fraud' or 'fraud_long'\n");
      return 1;
    }
  }

  if (argc > 2) {
    if (strcmp(argv[2], "probs") == 0) {
      output_type = PROBS;
    } else if (strcmp(argv[2], "binary") == 0) {
      output_type = BINARY;
    } else {
      fprintf(stderr, "Invalid output type. Use 'probs' or 'binary'\n");
      return 1;
    }
  }

  size_t NUMBER_OF_LAYERS = 0;
  size_t *layer_sizes = NULL;
  ActivationFunc *activations = NULL;
  ActivationFunc *activation_derivatives = NULL;

  if (strcmp(dataset_name, "fraud") == 0 ||
      (strcmp(dataset_name, "fraud_long") == 0)) {
    NUMBER_OF_LAYERS = 3;

    activations = malloc((NUMBER_OF_LAYERS - 2) * sizeof(ActivationFunc));
    activation_derivatives =
        malloc((NUMBER_OF_LAYERS - 2) * sizeof(ActivationFunc));
    layer_sizes = malloc(NUMBER_OF_LAYERS * sizeof(size_t));

    /*activations[0] = relu;*/
    /*activation_derivatives[0] = relu_derivative;*/
    activations[0] = heaviside_step_function;
    activation_derivatives[0] = heaviside_step_function_derivate;

    if (output_type == BINARY) {
      layer_sizes[0] = 2;
      layer_sizes[1] = 2;
      layer_sizes[2] = 1;
    } else if (output_type == PROBS) {
      layer_sizes[0] = 2;
      layer_sizes[1] = 2;
      layer_sizes[2] = 2;
    }
  }

  double ***initial_values = NULL;
  bool **frozen_neurons = NULL;
  if (argc > 3 && strcmp(argv[3], "true") == 0) {

    activations[0] = heaviside_step_function;
    activation_derivatives[0] = heaviside_step_function_derivate;

    initial_values = malloc(2 * sizeof(double **)); // Two non-input layers

    // Setting hidden layer initial values
    initial_values[0] = malloc(2 * sizeof(double *));
    initial_values[0][0] = malloc(3 * sizeof(double));
    initial_values[0][1] = malloc(3 * sizeof(double));

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        initial_values[0][i][j] = (double)random() / RAND_MAX - 0.5;
      }
    }

    initial_values[0][0][0] = -1.0;
    initial_values[0][0][1] = 0.0;
    initial_values[0][0][2] = 6;

    initial_values[0][1][0] = 0.0;
    initial_values[0][1][1] = 1.0;
    initial_values[0][1][2] = -10.0;

    // Setting output layer initial values
    initial_values[1] = malloc(1 * sizeof(double *));  // one neuron
    initial_values[1][0] = malloc(3 * sizeof(double)); // two inputs + one bias

    initial_values[1][0][0] = 1.0;
    initial_values[1][0][1] = 1.0;
    initial_values[1][0][2] = -1.5;

    // Setting frozen neurons
    frozen_neurons = malloc(2 * sizeof(bool *)); // Two non-input layers

    // Hidden layer not frozen
    frozen_neurons[0] = malloc(2 * sizeof(bool));
    frozen_neurons[0][0] = true;
    frozen_neurons[0][1] = true;

    // Output layer frozen
    frozen_neurons[1] = malloc(1 * sizeof(bool));
    frozen_neurons[1][0] = true;
  }

  // You can also change the following for other execution parameters
  const size_t epochs = 1000;
  const float learning_rate = 0.0001f;

  LayerActivationFunc *layer_activation_func;
  if (output_type == PROBS) {
    layer_activation_func = softmax;
  } else if (output_type == BINARY) {
    layer_activation_func = layer_heaviside_step_function;
  }

  NeuralNetwork *network = create_neural_network(
      layer_sizes, NUMBER_OF_LAYERS, activations, activation_derivatives,
      initial_values, frozen_neurons);

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

  print_network_weights(network);
  printf("\nTraining completed. Final results:\n");
  float accuracy = calculate_accuracy(network, dataset, num_test_cases,
                                      layer_activation_func);
  printf("\nFinal accuracy: %.2f%%\n", accuracy * 100);

  free_network_gradients(gradients);
  free_neural_network(network);
  free(layer_sizes);
  free(activations);
  free(activation_derivatives);

  if (initial_values != NULL) {
    for (int i = 0; i < 2; i++) {
      if (initial_values[i] != NULL) {
        for (int j = 0; j < (i == 0 ? 2 : 1); j++) {
          free(initial_values[i][j]);
        }
        free(initial_values[i]);
      }
    }
    free(initial_values);
  }

  if (frozen_neurons != NULL) {
    for (int i = 0; i < 2; i++) {
      free(frozen_neurons[i]);
    }
    free(frozen_neurons);
  }

  return 0;
}
