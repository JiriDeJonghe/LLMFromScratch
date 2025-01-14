#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "types.h"

typedef float (*ActivationFunc)(float);

/**
* @brief Evaluates the sigmoid function for a value
*
* @param value Value to evaluate the sigmoid for
*/
float sigmoid(float value);


/**
* @brief Evaluates the derivative of the sigmoid function for a value
*
* @param value Value to evaluate the derivative of the sigmoid for
*/
float sigmoid_derivative(float value);


/**
* @brief Evaluates the ReLU function for a value
*
* @param value Value to evaluate the ReLU for
*/
float relu(float value);


/**
* @brief Evaluates the derivative of the ReLU function for a value
*
* @param value Value to evaluate the derivative of the ReLU for
*/
float relu_derivative(float value);


/**
* @brief Applies the softmax function to a layer. Applying the softmax will turn the outputs into probabilities
*
* @param layer Layer to apply the softmax to
*/
void softmax(Layer* layer);

#endif
