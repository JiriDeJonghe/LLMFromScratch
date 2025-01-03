#ifndef PERCEPTRON_H
#define PERCEPTRON_H

/**
* @struct Perceptron
* @brief Represents a single perceptron
*/
typedef struct Perceptron {
    float* weights; // Array of input weights
    float bias; // Bias term. This is bias term that gets added to your perceptron, independent of the weights.
    float output; // The calculated output of the perceptron. This is a weighted sum using the weights and the inputs.
    int num_inputs; // The number of input connections. This is the same number of weights.
} Perceptron;

/**
* @struct Sample
* @brief Represents a single sample to be fitted by the perceptron
*/
typedef struct Sample {
    float* inputs; // Array of inputs for this sample.
    int output; // The traget label for this sample.
} Sample;

/**
* @struct Dataset
* @brief Represents the dataset on which to fit the perceptron on. Consists of Samples
*/
typedef struct Dataset {
    Sample* samples; // Array of samples in this dataset.
    int num_samples; // The number of samples in this dataset.
} Dataset;

/**
* @struct TrainingParameters
* @brief Combines all the possible training parameters in a single struct
*/
typedef struct TrainingParameters {
    float learning_rate; // The learning rate to be used in each training step.
    int num_training_steps; // The number of training steps to perform.
} TrainingParameters;

Dataset* create_dataset(int num_samples, int num_inputs);

#endif
