#ifndef GANN_H
#define GANN_H

// --- Main header for the Genetic Algorithm Neural Network (GANN) library ---

// --- Low-Level API ---

// Include all the public headers of the library for convenience.
// These must come first so the types are defined for the high-level API.
#include "data_loader.h"
#include "evolution.h"
#include "neural_network.h"
#include "backpropagation.h"
#include "selection.h"
#include "crossover.h"
#include "mutation.h"
#include "gann_errors.h" // Include the new error handling header
#include <stdbool.h>


// --- High-Level "Easy" API ---

/**
 * @brief Parameters for the gann_train function.
 *
 * This struct holds all the parameters needed to configure the training process
 * for the genetic algorithm neural network.
 */
typedef struct {
    const int* architecture;        /**< An array defining the number of neurons in each layer. */
    int num_layers;                 /**< The total number of layers in the network. */
    int population_size;            /**< The number of networks in each generation. */
    int num_generations;            /**< The number of generations to run the evolution for. */
    float mutation_rate;            /**< The magnitude of mutations when they occur. */
    float mutation_chance;          /**< The probability of a mutation occurring on any given weight or bias. */
    int fitness_samples;            /**< The number of samples from the dataset to use for fitness evaluation in each generation. Use 0 for the full dataset. */
    SelectionType selection_type;   /**< The method for selecting the fittest individuals (e.g., ELITE, TOURNAMENT). */
    int tournament_size;            /**< The number of individuals to compete in a tournament, if tournament selection is used. */
    ActivationType activation_hidden; /**< The activation function to use for the hidden layers. */
    ActivationType activation_output; /**< The activation function to use for the output layer. */
    CrossoverType crossover_type;   /**< The crossover strategy to use. */
    MutationType mutation_type;     /**< The mutation strategy to use. */
    double mutation_std_dev;        /**< The standard deviation for Gaussian mutation. */
    bool logging;                   /**< Whether to print logging information during training. */
} GannTrainParams;

/**
 * @brief Creates a `GannTrainParams` struct with sensible default values.
 *
 * This function is a convenient way to get started with training without having
 * to manually set every parameter. The user must still set the `architecture`
 * and `num_layers` fields.
 *
 * @return A `GannTrainParams` struct with default values.
 */
GannTrainParams gann_create_default_params(void);


// --- Function Pointer Typedefs for Extensibility ---
typedef NetworkFitness* (*SelectionFunction)(NetworkFitness*, int, int*, SelectionType, int);
typedef NeuralNetwork* (*CrossoverFunction)(const NeuralNetwork*, const NeuralNetwork*, CrossoverType);
typedef void (*MutationFunction)(NeuralNetwork*, float, float, MutationType, double, int, int, double);


/**
 * @brief A struct for the new `gann_evolve` function, which allows for custom genetic operators.
 */
typedef struct {
    GannTrainParams base_params;
    SelectionFunction selection_func;
    CrossoverFunction crossover_func;
    MutationFunction mutation_func;
} GannEvolveParams;


/**
 * @brief Evolves a population of neural networks using the given genetic operators.
 *
 * This is a more flexible version of `gann_train` that allows for custom genetic operators.
 *
 * @param params The evolution parameters, including function pointers to the genetic operators.
 * @param train_dataset The dataset to train on.
 * @return A pointer to the best trained NeuralNetwork. The caller is responsible for freeing this network.
 *         Returns NULL on failure. If NULL is returned, call `gann_get_last_error()` to get the specific error code.
 */
NeuralNetwork* gann_evolve(const GannEvolveParams* params, const Dataset* train_dataset);


/**
 * @brief Trains a new neural network using a genetic algorithm.
 *
 * This function encapsulates the entire training loop.
 *
 * @param params The training parameters.
 * @param train_dataset The dataset to train on.
 * @return A pointer to the best trained NeuralNetwork. The caller is responsible for freeing this network.
 *         Returns NULL on failure. If NULL is returned, call `gann_get_last_error()` to get the specific error code.
 */
NeuralNetwork* gann_train(const GannTrainParams* params, const Dataset* train_dataset);



/**
 * @brief Trains a new neural network using backpropagation.
 *
 * @param params The backpropagation training parameters.
 * @param train_dataset The dataset to train on.
 * @return A pointer to the trained NeuralNetwork. The caller is responsible for freeing this network.
 *         Returns NULL on failure. If NULL is returned, call `gann_get_last_error()` to get the specific error code.
 */
NeuralNetwork* gann_train_with_backprop(const GannBackpropParams* params, const Dataset* train_dataset);


/**
 * @brief Makes a prediction on a single input vector.
 *
 * @param net The trained neural network.
 * @param input A flat array of input data (e.g., pixel values). Must match the network's input layer size.
 * @return The index of the predicted class (e.g., the digit 0-9), or -1 on failure.
 *         If -1 is returned, call `gann_get_last_error()` to get the specific error code.
 */
int gann_predict(const NeuralNetwork* net, const double* input);

/**
 * @brief Evaluates the network's accuracy on a given dataset.
 *
 * @param net The trained neural network.
 * @param dataset The dataset to evaluate on (e.g., a test set).
 * @return The accuracy of the network on the dataset (a value from 0.0 to 1.0).
 *         On failure, returns 0.0 and sets an error code. Call `gann_get_last_error()` to check for errors.
 */
double gann_evaluate(const NeuralNetwork* net, const Dataset* dataset);

#endif // GANN_H
