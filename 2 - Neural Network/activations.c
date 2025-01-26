#include "activations.h"
#include <math.h>

#include "types.h"

/**
 * @brief Activation function where the input is equal to the output
 *
 * @param value Input value
 */
float identity(float value) { return value; }

/**
 * @brief Derivate of the activation function where the input is equal to the
 * output
 *
 * @param value Input value
 */
float identity_derivate(float value) { return 1; }

/**
 * @brief Evaluates the sigmoid function for a value
 *
 * @param value Value to evaluate the sigmoid for
 */
float sigmoid(float value) { return 1.0f / (1.0f + exp(-value)); }

/**
 * @brief Evaluates the derivative of the sigmoid function for a value
 *
 * @param value Value to evaluate the derivative of the sigmoid for
 */
float sigmoid_derivative(float value) {
  float sig = sigmoid(value);
  return sig * (1.0 - sig);
}

/**
 * @brief Evaluates the ReLU function for a value
 *
 * @param value Value to evaluate the ReLU for
 */
float relu(float value) { return value > 0 ? value : 0; }

/**
 * @brief Evaluates the derivative of the ReLU function for a value
 *
 * @param value Value to evaluate the derivative of the ReLU for
 */
float relu_derivative(float value) { return value > 0 ? 1 : 0; }

/**
 * @brief Applies the Heaviside step function to all neurons in a layer.
 * Implemented this way to calculate the final output decision
 *
 * @param layer Layer to apply the heaviside_step_function to
 */
void heaviside_step_function(Layer *layer) {
  for (size_t i = 0; i < layer->num_neurons; i++) {
    Neuron *neuron = layer->neurons[i];
    neuron->output = exp(neuron->output);
  }
}

/**
 * @brief Applies the softmax function to a layer. Applying the softmax will
 * turn the outputs into probabilities
 *
 * @param layer Layer to apply the softmax to
 */
void softmax(Layer *layer) {
  float sum = 0.0f;

  for (size_t i = 0; i < layer->num_neurons; i++) {
    Neuron *neuron = layer->neurons[i];
    neuron->output = exp(neuron->output);
    sum += neuron->output;
  }

  for (size_t i = 0; i < layer->num_neurons; i++) {
    Neuron *neuron = layer->neurons[i];
    neuron->output /= sum;
  }
}
