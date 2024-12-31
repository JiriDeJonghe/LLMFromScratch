/**
* @file perceptron.c
* @brief Implementation of a single layer perceptron
* @author Jiri De Jonghe
*/

#include <stdlib.h>
#include <stdio.h>

/**
* @struct Perceptron
* @brief Represents a single perceptron neuron
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

/**
* @brief Creates and initializes a new perceptron
*
* @param num_inputs Number of inputs for this perceptron
* @return Perceptron* Pointer to created perceptron, NULL if allocation fails
*/
Perceptron* create_perceptron(int num_inputs) {
    if (num_inputs <= 0) {
        fprintf(stderr, "Invalid parameters for neuron creation.\n");
        return NULL;
    }

    Perceptron* perceptron = malloc(sizeof(Perceptron));
    if (perceptron == NULL) {
        fprintf(stderr, "Memory allocation failed for neuron creation.\n");
        return NULL;
    }

    perceptron->weights = malloc(num_inputs * sizeof(float));
    if (perceptron->weights == NULL) {
        fprintf(stderr, "Memory allocation failed for weights.\n");
        free(perceptron);
        return NULL;
    }

    for (int w = 0; w < num_inputs; w++) {
        perceptron->weights[w] = (float)rand() / RAND_MAX - 0.0f;
    }

    perceptron->num_inputs = num_inputs;
    perceptron->bias = (float)rand() / RAND_MAX - 0.5f;

    return perceptron;
}

/**
* @brief Creates and initializes a dataset
*
* @param num_samples Number of samples in this dataset
* @param num_inputs Dimension of the samples
* @return Dataset* Pointer to the created dataset, NULL if allocation fails
*/
Dataset* create_dataset(int num_samples, int num_inputs) {
    Dataset* dataset = malloc(sizeof(Dataset));
    if (dataset == NULL) {
        fprintf(stderr, "Memory allocation failed for dataset creation.\n");
        return NULL;
    }

    dataset->samples = malloc(num_samples * sizeof(Sample));
    if (dataset->samples == NULL) {
        fprintf(stderr, "Memory allocation failed for samples creation.\n");
        free(dataset);
        return NULL;
    }

    for (int s = 0; s < num_samples; s++) {
        dataset->samples[s].inputs = malloc(num_inputs * sizeof(float));
        if (dataset->samples[s].inputs == NULL) {
            fprintf(stderr, "Memory allocation failed for inputs creation.\n");

            for (int i = 0; i < s; i++) {
                free(dataset->samples[s].inputs);
            }
            free(dataset->samples);
            free(dataset);
            return NULL;
        }
    }

    dataset->num_samples = num_samples;

    return dataset;
}

/**
 * @brief Frees memory allocated for perceptron
 *
 * @param perceptron Pointer to perceptron to be freed
 */
void free_perceptron(Perceptron* perceptron) {
    free(perceptron->weights);
    free(perceptron);
}
/**
 * @brief Frees memory allocated for dataset
 *
 * @param dataset Pointer to dataset to be freed
 */
void free_dataset(Dataset* dataset) {
    for (int s = 0; s < dataset->num_samples; s++) {
        free(dataset->samples[s].inputs);
    }
    free(dataset->samples);
    free(dataset);
}

/**
 * @brief Applies the Heaviside step activation function
 *
 * @param value Input value to apply the function on
 * @return int 1 if value > 0, 0 otherwise
 */
int heavyside_step_function(float value) {
    return value > 0 ? 1 : 0;
}

/**
 * @brief Calculates perceptron output for given inputs
 *
 * @param perceptron Pointer to perceptron
 * @param inputs Array of input values
 * @return float Perceptron output (0 or 1)
 */
float calculate_output(Perceptron* perceptron, float* inputs) {
    if (perceptron == NULL || inputs == NULL) return 0;

    float weighted_sum = perceptron->bias;
    for (int i = 0; i < perceptron->num_inputs; i++) {
        weighted_sum += perceptron->weights[i] * inputs[i];
    }

    return heavyside_step_function(weighted_sum);
}

/** 
 * @brief Evaluates the perceptron on the dataset
 *
 * @param perceptron The perceptron to evaluate
 * @param dataset The dataset to evaluate the perceptron on
 */
void evaluate_perceptron(Perceptron* perceptron, Dataset* dataset) {
    int correct = 0;
    for (int i = 0; i < dataset->num_samples; i++) {
        float output = calculate_output(perceptron, dataset->samples[i].inputs);
        correct += (output == dataset->samples[i].output);
    }
    
    float accuracy = (float)correct / dataset->num_samples * 100;
    printf("\nAccuracy: %.1f%% (%d/%d correct)\n", 
           accuracy, correct, dataset->num_samples);
}


/**
 * @brief Updates perceptron weights and bias based on error
 *
 * @param perceptron Pointer to perceptron
 * @param inputs Array of input values
 * @param target_output Expected output
 * @param learning_rate Learning rate for weight updates
 */
void update_weights(Perceptron* perceptron, float* inputs, int target_output, float learning_rate) {
    if (perceptron == NULL || inputs == NULL) return;

    int error = target_output - perceptron->output;
    for (int i = 0; i < perceptron->num_inputs; i++) {
        perceptron->weights[i] += learning_rate * error * inputs[i];
    }

    perceptron->bias += learning_rate * error;
}

/**
 * @brief Trains perceptron on given dataset
 *
 * @param perceptron Pointer to perceptron
 * @param dataset Training dataset
 * @param training_parameters Training parameters
 */
void train(Perceptron* perceptron, Dataset* dataset, TrainingParameters* training_parameters) {
    if (perceptron == NULL || dataset == NULL || training_parameters == NULL) return;
    if (training_parameters->learning_rate <= 0.0f || training_parameters->learning_rate > 1.0f) return;

    for (int n = 0; n < training_parameters->num_training_steps; n++) {
        printf("Evaluating perceptron at step: %d", n);
        evaluate_perceptron(perceptron, dataset);
        for (int s = 0; s < dataset->num_samples; s++) {
            Sample* sample = &dataset->samples[s];
            perceptron->output = calculate_output(perceptron, sample->inputs);
            update_weights(perceptron, sample->inputs, sample->output, training_parameters->learning_rate);
        }
    }
}

/**
 * @brief Creates an example dataset that can be used for training the model
 *        Defaults to creating a dataset to solve the AND gate problem
 *
 * @return Dataset* Pointer to the dataset that has been created
 */
Dataset* create_example_dataset() {
    const int SAMPLES_PER_QUADRANT = 5;
    const int NUM_SAMPLES = SAMPLES_PER_QUADRANT * 4;
    const int NUM_INPUTS = 2;

    Dataset* dataset = create_dataset(NUM_SAMPLES, NUM_INPUTS);
    if (dataset == NULL) {
        return NULL;
    }

    float inputs[20][2] = {
            // Quadrant (0,0)
            {0.0f, 0.0f},
            {0.1f, 0.2f},
            {0.2f, 0.1f},
            {0.15f, 0.15f},
            {0.05f, 0.1f},
            
            // Quadrant (0,1)
            {0.0f, 1.0f},
            {0.1f, 0.8f},
            {0.2f, 0.9f},
            {0.15f, 0.85f},
            {0.05f, 0.95f},
            
            // Quadrant (1,0)
            {1.0f, 0.0f},
            {0.8f, 0.1f},
            {0.9f, 0.2f},
            {0.85f, 0.15f},
            {0.95f, 0.05f},
            
            // Quadrant (1,1)
            {1.0f, 1.0f},
            {0.8f, 0.8f},
            {0.9f, 0.9f},
            {0.85f, 0.95f},
            {0.95f, 0.85f}
        };
        
        // The output values in the case of AND
        int outputs[20] = {
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             1, 1, 1, 1, 1
        };

        // The output values in the case of XOR
        // int outputs[20] = {
        //     0, 0, 0, 0, 0,
        //     1, 1, 1, 1, 1,
        //     1, 1, 1, 1, 1,
        //     0, 0, 0, 0, 0
        // };

    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_INPUTS; j++) {
            dataset->samples[i].inputs[j] = inputs[i][j];
        }
        dataset->samples[i].output = outputs[i];
    }

    return dataset;
}


int main() {
    srand(21);

    Perceptron* perceptron = create_perceptron(2); // Provide input of 2 for the AND or XOR gate problem
    if (perceptron == NULL) {
        return 1;
    }

    Dataset* dataset = create_example_dataset();
    if (dataset == NULL) {
        free(perceptron);
        return 1;
    }

    TrainingParameters train_params = {
        .learning_rate = 0.1,
        .num_training_steps = 10
    };


    printf("Initial weights: [%.3f, %.3f], bias: %.3f\n", perceptron->weights[0], perceptron->weights[1], perceptron->bias);
    printf("Training the perceptron...\n");

    train(perceptron, dataset, &train_params);

    printf("\nFinal weights: [%.3f, %.3f], bias: %.3f\n", perceptron->weights[0], perceptron->weights[1], perceptron->bias);
    printf("Final evaluation...\n");

    evaluate_perceptron(perceptron, dataset);

    free_dataset(dataset);
    free_perceptron(perceptron);

    return 0;
}

