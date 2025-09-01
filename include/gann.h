#ifndef GANN_H
#define GANN_H

// --- Main header for the Genetic Algorithm Neural Network (GANN) library ---

// --- Low-Level API ---

// Include all the public headers of the library for convenience.
// These must come first so the types are defined for the high-level API.
#include "data_loader.h"
#include "evolution.h"
#include "neural_network.h"


// --- High-Level "Easy" API ---

/**
 * @brief Parameters for the gann_train function.
 */
typedef struct {
    const int* architecture;
    int num_layers;
    int population_size;
    int num_generations;
    float mutation_rate;
    float mutation_chance;
    int fitness_samples; // Number of samples from the dataset to use for fitness evaluation
    SelectionType selection_type;
    int tournament_size;
} GannTrainParams;

/**
 * @brief Trains a new neural network using a genetic algorithm.
 *
 * This function encapsulates the entire training loop.
 *
 * @param params The training parameters.
 * @param train_dataset The dataset to train on.
 * @return A pointer to the best trained NeuralNetwork. The caller is responsible for freeing this network.
 */
NeuralNetwork* gann_train(const GannTrainParams* params, const Dataset* train_dataset);

/**
 * @brief Makes a prediction on a single input vector.
 *
 * @param net The trained neural network.
 * @param input A flat array of input data (e.g., pixel values). Must match the network's input layer size.
 * @return The index of the predicted class (e.g., the digit 0-9).
 */
int gann_predict(const NeuralNetwork* net, const double* input);

/**
 * @brief Evaluates the network's accuracy on a given dataset.
 *
 * @param net The trained neural network.
 * @param dataset The dataset to evaluate on (e.g., a test set).
 * @return The accuracy of the network on the dataset (a value from 0.0 to 1.0).
 */
double gann_evaluate(const NeuralNetwork* net, const Dataset* dataset);

#endif // GANN_H
