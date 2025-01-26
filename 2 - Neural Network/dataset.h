#ifndef DATASET_H
#define DATASET_H

/**
 * @struct Sample
 * @brief Represents a sample on which to train the neural network on
 */
typedef struct {
  float *inputs;
  float *expected_outputs;
} Sample;

/**
 * @struct Dataset
 * @brief Represents the dataset on which to train the neural network on.
 * Consists of Samples
 */
typedef struct {
  Sample *samples; // Array of samples in this dataset.
  int num_samples; // The number of samples in this dataset.
} Dataset;

Dataset *create_example_dataset(char *dataset_name);

#endif
