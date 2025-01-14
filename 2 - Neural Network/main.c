#include <stdio.h>
#include <math.h>

#include "activations.h"
#include "types.h"


// Loss Calculation - Cross Entropy Loss and its derivate
float calculate_cross_entropy_loss(float* predictions, float* truths, size_t num_classes) {
    float loss = 0.0f;
    for (size_t i = 0; i < num_classes; i++) {
        loss -= truths[i] * logf(predictions[i] + 1e-10f);
    }
    return loss;
}

/**
* @brief Calculates a forward pass of the neural network. The final results can be found by inspecting the final layers
*
* @param network The neural network to calculate the forward pass for
* @param input The input to calculate the forward pass for
*/
void forward_propagate(NeuralNetwork* network, const float* input) {
    if (network == NULL || input == NULL) {
        fprintf(stderr, "Invalid parameters for forward propagation.\n");
        return;
    }

    // Setting input layer: each node is assigned to the input
    Layer* input_layer = network->layers[0];
    for (size_t i = 0; i < input_layer->num_neurons; i++) {
        input_layer->neurons[i]->output = input[i];
    }

    // Calculating subsequent layers:
    //  Iterate over the layers from Left to Right (with Left being the input, and Right being the output layer)
    //  Determine the output of each neuron by getting the weighted sum (with bias) and applying the activation function
    for (size_t layer_id = 1; layer_id < network->num_layers; layer_id++) {
        Layer *current_layer = network->layers[layer_id];
        Layer *previous_layer = network->layers[layer_id - 1];

        for (size_t neuron_id = 0; neuron_id < current_layer->num_neurons;
        neuron_id++) {
            Neuron *current_neuron = current_layer->neurons[neuron_id];

            float weighted_sum = current_neuron->bias;
            for (size_t prev_neuron_id = 0; prev_neuron_id < previous_layer->num_neurons;
            prev_neuron_id++) {
                Neuron *prev_neuron = previous_layer->neurons[prev_neuron_id];
                weighted_sum +=
                    prev_neuron->output * current_neuron->weights[prev_neuron_id];
            }

            if (layer_id != network->num_layers - 1) {
                ActivationFunc activation = current_layer->activation;
                current_neuron->output = activation(weighted_sum);
            } else {
                current_neuron->output = weighted_sum; // Store raw value for softmax in last layer to get probabilities
            }
        }

        if (layer_id == network->num_layers - 1) {
            softmax(current_layer);
        }
    }
}

/**
* @brief Calculates a backward pass of the neural network. The weights and biases will be updated accordingly.
*
* @param network The neural network to calculate the backpropagation for
* @param gradients The gradients of the neural network
* @param input The input for this backpropagation
* @param result The true result value for the inputs
* @param learning_rate The learning rate to use for this backward pass
*/
void backward_propagate(NeuralNetwork* network, NetworkGradients* gradients, const float* input, const float* result, const float learning_rate) { 
    if (network == NULL || gradients == NULL || input == NULL || result == NULL || learning_rate < 0 || learning_rate > 1) {
        fprintf(stderr, "Invalid parameters for backward propagation.\n");
        return;
    }

    // Clear Gradients
    for (size_t i = 0; i < network->num_layers; i++) {
        Layer* layer = network->layers[i];
        for (size_t j = 0; j < layer->num_neurons; j++) {
            Neuron* neuron = layer->neurons[j];
            for (size_t k = 0; k < neuron->num_inputs; k++) {
                gradients->layer_gradients[i]->neuron_gradients[j]->weight_gradients[k] = 0.0f;
            }
            gradients->layer_gradients[i]->neuron_gradients[j]->error = 0.0f;
        }
    }

    // First handle the output layer
    size_t output_layer_idx = network->num_layers - 1;
    Layer* output_layer = network->layers[output_layer_idx];
    Layer* prev_layer = network->layers[output_layer_idx - 1];
    for (size_t i = 0; i < output_layer->num_neurons; i++) {
        // Calculate the error at the output layer:
        // We define the error as follows: E_0 = C'(y_pred) * R'(Z_0)
        // With C the cost function, R the activation function for the output layer and Z_0 the weighted sums of the nodes of the output layer
        // For cross entropy after soft max this is: y_pred - y_true
        float error = output_layer->neurons[i]->output - result[i];

        // After that we calculate the derivative of the Cost with respect to the edges incoming into the nodes of the output layer.
        // The gradient for each of these edges is defined as w_ij = error * o_j
        for (size_t j = 0; j < output_layer->neurons[i]->num_inputs; j++) {
            gradients->layer_gradients[output_layer_idx]->neuron_gradients[i]->weight_gradients[j] = error * prev_layer->neurons[j]->output;
        }
        gradients->layer_gradients[output_layer_idx]->neuron_gradients[i]->error = error;
    }

    // Now that we have defined the error and the gradients for the last layer, we propagate it through the network
    // Calculate the gradients of the hidden layers: E_h = E_(l+1) * W_(l+1) * R'(Z_h)
    for (int i = (int)network->num_layers - 2; i >= 0; i--) {
        Layer* layer = network->layers[i];
        Layer* next_layer = network->layers[i+1];

        LayerGradients* layer_gradients = gradients->layer_gradients[i];
        LayerGradients* next_layer_gradients = gradients->layer_gradients[i+1];

        // Calculate the error and gradients for each neuron in the current layer
        for (size_t j = 0; j < layer->num_neurons; j++) {
            float error = 0.0f;
            for (size_t next_j = 0; next_j < next_layer->num_neurons; next_j++) {
                float next_error = next_layer_gradients->neuron_gradients[next_j]->error;
                error += next_layer->neurons[next_j]->weights[j] * next_error;
            }
            error *= layer->activation_derivative(layer->neurons[j]->output);
            layer_gradients->neuron_gradients[j]->error = error;

            // Calculating gradients 
            if (i > 0) {
                Layer* prev_layer = network->layers[i - 1];
                for (size_t w = 0; w < layer->neurons[j]->num_inputs; w++) {
                    layer_gradients->neuron_gradients[j]->weight_gradients[w] = error * prev_layer->neurons[w]->output;
                }
            } else {
                for (size_t w = 0; w < layer->neurons[j]->num_inputs; w++) {
                    layer_gradients->neuron_gradients[j]->weight_gradients[w] = error * input[w];
                }
            }
        }
    }
    
    for (size_t l = 0; l < network->num_layers; l++) {
        Layer* layer = network->layers[l];
        for (size_t n = 0; n < layer->num_neurons; n++) {
            Neuron* neuron = layer->neurons[n];
            for (size_t w = 0; w < neuron->num_inputs; w++) {
                neuron->weights[w] -= learning_rate * gradients->layer_gradients[l]->neuron_gradients[n]->weight_gradients[w];
            }
            neuron->bias -= learning_rate * gradients->layer_gradients[l]->neuron_gradients[n]->error;
        }
    }
}


void print_output_layer(NeuralNetwork *network) {
    Layer *output_layer = network->layers[network->num_layers - 1];

    printf("Network Output:\n");
    for (size_t i = 0; i < output_layer->num_neurons; i++) {
        printf("Neuron %zd: %f\n", i, output_layer->neurons[i]->output);
    }
}
