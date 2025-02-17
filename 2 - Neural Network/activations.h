#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "types.h"

typedef float (*ActivationFunc)(float);
typedef void(LayerActivationFunc)(Layer *);

/**
 * @brief Activation function where the input is equal to the output
 *
 * @param value Input value
 */
float identity(float value);

/**
 * @brief Derivate of the activation function where the input is equal to the
 * output
 *
 * @param value Input value
 */
float identity_derivate(float value);

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
 * @brief Applies the Heaviside step function to a value
 *
 * @param value Value to evaluate the heaviside step function for
 */
float heaviside_step_function(float value);

/**
 * @brief Applies the derivate of the Heaviside step function to a value
 * In reality you can't use a non-differentiable function for backpropagation.
 * We make a simplification to learn anyway
 *
 * @param value Value to evaluate the derivative of the heaviside step function
 * for
 */
float heaviside_step_function_derivate(float value);

/**
 * @brief Applies the Heaviside step function to all neurons in a layer.
 * Implemented this way to calculate the final output decision
 *
 * @param layer Layer to apply the heaviside_step_function to
 */
void layer_heaviside_step_function(Layer *layer);

/**
 * @brief Applies the softmax function to a layer. Applying the softmax will
 * turn the outputs into probabilities
 *
 * @param layer Layer to apply the softmax to
 */
void softmax(Layer *layer);

#endif
