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
    CrossoverType crossover_type;   /**< The crossover strategy to use. */
    MutationType mutation_type;     /**< The mutation strategy to use. */
    double mutation_std_dev;        /**< The standard deviation for Gaussian mutation. */
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
 * @brief Trains a new neural network using backpropagation.
 *
 * @param params The backpropagation training parameters.
 * @param train_dataset The dataset to train on.
 * @return A pointer to the trained NeuralNetwork. The caller is responsible for freeing this network.
 */
NeuralNetwork* gann_train_with_backprop(const GannBackpropParams* params, const Dataset* train_dataset);


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
