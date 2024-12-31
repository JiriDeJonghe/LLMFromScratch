#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "types.h"

typedef float (*ActivationFunc)(float);

float sigmoid(float value);
float sigmoid_derivative(float value);
float relu(float value);
float relu_derivative(float value);
void softmax(Layer* layer);

#endif
