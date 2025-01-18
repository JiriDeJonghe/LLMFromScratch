#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "network.h"
#include "propagation.h"

typedef struct {
    float* inputs;
    float* expected_outputs;
} TestCase;

TestCase xor_test_cases[] = {
    {(float[]){0, 0}, (float[]){0, 1}},
    {(float[]){0, 1}, (float[]){1, 0}},
    {(float[]){1, 0}, (float[]){1, 0}},
    {(float[]){1, 1}, (float[]){0, 1}}
};

float calculate_accuracy(NeuralNetwork* network, TestCase* test_cases, size_t num_cases) {
    int correct = 0;

    for (size_t i = 0; i < num_cases; i++) {
        forward_propagate(network, test_cases[i].inputs);

        Layer* output_layer = network->layers[network->num_layers - 1];
        float prediction = output_layer->neurons[0]->output;
        float expected = test_cases[i].expected_outputs[0];

        if ((prediction >= 0.5 && expected == 1) || (prediction < 0.5 && expected == 0)) {
            correct++;
        }

        printf("Input: [%.0f, %.0f], Expected: %.0f, Predicted: %.2f\n",
               test_cases[i].inputs[0], test_cases[i].inputs[1], 
               expected, prediction);
    }

    return (float)correct / num_cases;
}


int main() {
    size_t layer_sizes[] = {2, 4, 2};
    ActivationFunc activations[] = {relu, relu};
    ActivationFunc activation_derivatives[] = {relu_derivative, relu_derivative};

    NeuralNetwork* network = create_neural_network(layer_sizes, 3, activations, activation_derivatives);
    if (network == NULL) {
        fprintf(stderr, "Failed to create neural network\n");
        return 1;
    }

    NetworkGradients* gradients = create_network_gradients(network);
    if (gradients == NULL) {
        fprintf(stderr, "Failed to create gradients\n");
        free_neural_network(network);
        return 1;
    }

    const size_t epochs = 1000;
    const float learning_rate = 0.01f;
    const size_t num_test_cases = sizeof(xor_test_cases) / sizeof(TestCase);

    printf("Training started...\n");
    forward_propagate(network, xor_test_cases[0].inputs);
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float total_error = 0.0f;

        for (size_t i = 0; i < num_test_cases; i++) {
            forward_propagate(network, xor_test_cases[i].inputs);
            backward_propagate(network, gradients, xor_test_cases[i].inputs,
                             xor_test_cases[i].expected_outputs, learning_rate);

            Layer* output_layer = network->layers[network->num_layers - 1];
            float prediction = output_layer->neurons[0]->output;
            float expected = xor_test_cases[i].expected_outputs[0];
            total_error += (prediction - expected) * (prediction - expected);
        }


        if ((epoch + 1) % 10 == 0) {
            printf("Epoch %zu, Error: %.4f\n", epoch + 1, total_error / num_test_cases);
        }
    }

    printf("\nTraining completed. Final results:\n");
     float accuracy = calculate_accuracy(network, xor_test_cases, num_test_cases);
    printf("\nFinal accuracy: %.2f%%\n", accuracy * 100);

    free_network_gradients(gradients);
    free_neural_network(network);


    return 0;
}
