#include <stdio.h>
#include <math.h>

#include "activations.h"
#include "types.h"


// Loss Calculation - Cross Entropy Loss and its derivate
float calculate_cross_entropy_loss(float* predictions, float* truths, int num_classes) {
    float loss = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        loss -= truths[i] * logf(predictions[i] + 1e-10f);
    }
    return loss;
}

// Forward Propagation
void forward_propagate(NeuralNetwork* network, float* input) {
    // Setting input layer: each node is assigned to the input
    Layer* input_layer = &network->layers[0];
    for (int i = 0; i < input_layer->num_neurons; i++) {
        input_layer->neurons[i].output = input[i];
    }

    // Calculating subsequent layers:
    //  Iterate over the layers from Left to Right (with Left being the input, and Right being the output layer)
    //  Determine the output of each neuron by getting the weighted sum (with bias) and applying the activation func
    for (int layer_id = 1; layer_id < network->num_layers; layer_id++) {
        Layer *current_layer = &network->layers[layer_id];
        Layer *previous_layer = &network->layers[layer_id - 1];

        for (int neuron_id = 0; neuron_id < current_layer->num_neurons;
        neuron_id++) {
            Neuron *current_neuron = &current_layer->neurons[neuron_id];

            float weighted_sum = current_neuron->bias;
            for (int prev_neuron_id = 0; prev_neuron_id < previous_layer->num_neurons;
            prev_neuron_id++) {
                Neuron *prev_neuron = &previous_layer->neurons[prev_neuron_id];
                weighted_sum +=
                    prev_neuron->output * current_neuron->weights[prev_neuron_id];
            }

            if (layer_id == network->num_layers - 1) {
                softmax(current_layer);
            } else {
                ActivationFunc activation = current_neuron->activation;
                current_neuron->output = activation(weighted_sum);
            }
        }
    }
}

// Backward Propagation
void backward_propagate(NeuralNetwork* network, float* input, float* result, float learning_rate, NetworkGradients* gradients) { 
    // Clear Gradients
    for (int i = 0; i < network->num_layers; i++) {
        Layer* layer = &network->layers[i];
        for (int j = 0; j < layer->num_neurons; j++) {
            Neuron* neuron = &layer->neurons[j];
            for (int k = 0; k < neuron->num_inputs; k++) {
                gradients->layer_gradients[i].neuron_gradients[j].weight_gradients[k] = 0.0f;
            }
            gradients->layer_gradients[i].neuron_gradients[j].error = 0.0f;
        }
    }

    // First handle the output layer separately
    int output_layer_idx = network->num_layers - 1;
    Layer* output_layer = &network->layers[output_layer_idx];
    for (int i = 0; i < output_layer->num_neurons; i++) {
        // Calculate the error at the output layer:
        // We define the error as follows: E_0 = C'(y_pred) * R'(Z_0)
        // With C the cost function, R the activation function for the output layer and Z_0 the weighted sums of the nodes of the output layer
        // For cross entropy after soft max this is: y_pred - y_true
        float error = output_layer->neurons[i].output - result[i];

        // Take the error of the nodes of the next layer and multiply it with the weight to this node and the derivate of the activation function
        // Iterate over all the nodes of the previous layer to get the gradient for all the incoming nodes
        for (int j = 0; j < output_layer->neurons[i].num_inputs; j++) {
            Layer* prev_layer = &network->layers[output_layer_idx - 1];
            gradients->layer_gradients[output_layer_idx].neuron_gradients[i].weight_gradients[j] = error * prev_layer->neurons[j].output;
        }
        gradients->layer_gradients[output_layer_idx].neuron_gradients[i].error = error;
    }

    // Now that we have defined the error for the last layer, we propagate it through the network
    // Calculate the gradients of the hidden layers: E_h = E_(l+1) * W_(l+1) * R'(Z_h)
    for (int i = network->num_layers - 2; i >= 0; i--) {
        Layer* layer = &network->layers[i];
        Layer* next_layer = &network->layers[i+1];

        LayerGradients* layer_gradients = &gradients->layer_gradients[i];
        LayerGradients* next_layer_gradients = &gradients->layer_gradients[i+1];

        // Calculate the error and gradients for each neuron in the current layer
        for (int j = 0; i < layer->num_neurons; i++) {
            float error = 0.0f;
            for (int next_j = 0; next_j < next_layer->num_neurons; next_j++) {
                float next_error = next_layer_gradients->neuron_gradients[next_j].error;
                error += next_layer->neurons[next_j].weights[j] * next_error;
            }
            error *= layer->neurons[j].derivate_activation(layer->neurons[j].output);
            layer_gradients->neuron_gradients[j].error = error;

            // Calculating gradients 
            if (i > 0) {
                Layer* prev_layer = &network->layers[i - 1];
                for (int w = 0; w < layer->neurons[j].num_inputs; w++) {
                    layer_gradients->neuron_gradients[j].weight_gradients[w] = error * prev_layer->neurons[w].output;
                }
            } else {
                for (int w = 0; w < layer->neurons[j].num_inputs; w++) {
                    layer_gradients->neuron_gradients[j].weight_gradients[w] = error * input[w];
                }
            }
        }
    }
    
    for (int l = 0; l < network->num_layers; l++) {
        Layer* layer = &network->layers[l];
        for (int n = 0; n < layer->num_neurons; n++) {
            Neuron* neuron = &layer->neurons[n];
            for (int w = 0; w < neuron->num_inputs; w++) {
                neuron->weights[w] -= learning_rate * gradients->layer_gradients[l].neuron_gradients[n].weight_gradients[w];
            }
            neuron->bias -= learning_rate * gradients->layer_gradients[l].neuron_gradients[n].error;
        }
    }
}


void print_output_layer(NeuralNetwork *network) {
    Layer *output_layer = &network->layers[network->num_layers - 1];

    printf("Network Output:\n");
    for (int i = 0; i < output_layer->num_neurons; i++) {
        printf("Neuron %d: %f\n", i, output_layer->neurons[i].output);
    }
}
