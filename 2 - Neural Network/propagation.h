#ifndef PROPAGATION_H
#define PROPAGATION_H

#include <stdio.h>
#include "types.h"

float calculate_cross_entropy_loss(float* predictions, float* truths, size_t num_classes);

/**
* @brief Calculates a forward pass of the neural network. The final results can be found by inspecting the final layers
*
* @param network The neural network to calculate the forward pass for
* @param input The input to calculate the forward pass for
*/
void forward_propagate(NeuralNetwork* network, const float* input);

/**
* @brief Calculates a backward pass of the neural network. The weights and biases will be updated accordingly.
*
* @param network The neural network to calculate the backpropagation for
* @param gradients The gradients of the neural network
* @param input The input for this backpropagation
* @param result The true result value for the inputs
* @param learning_rate The learning rate to use for this backward pass
*/
void backward_propagate(NeuralNetwork* network, NetworkGradients* gradients, const float* input, const float* result, const float learning_rate);

#endif
