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
 * @brief Seeds the random number generator used by the library.
 * @details Call this function once at the beginning of your program to ensure
 * reproducible results from the training process, which relies on randomness
 * for weight initialization, mutations, and some selection/crossover methods.
 * @param seed The seed for the random number generator. A common practice is to
 * use a fixed integer for development and `time(NULL)` for production runs.
 */
void gann_seed_rng(unsigned int seed);

/**
 * @brief Parameters for training a neural network with a genetic algorithm.
 * @details This struct holds all the parameters needed to configure the
 * training process. Use `gann_create_default_params()` to get a struct with
 * sensible default values, and then override fields as needed.
 */
typedef struct {
    const int* architecture;        /**< An array defining the number of neurons in each layer, e.g., `{784, 128, 10}`. */
    int num_layers;                 /**< The total number of layers in the network (size of the `architecture` array). */
    int population_size;            /**< The number of neural networks in each generation's population. */
    int num_generations;            /**< The maximum number of generations to run the evolution for. */
    float mutation_rate;            /**< The magnitude of change applied during mutation. For Gaussian mutation, this is the standard deviation. */
    float mutation_chance;          /**< The probability (0.0 to 1.0) of a mutation occurring on any given weight or bias. */
    int fitness_samples;            /**< The number of samples from the training dataset to use for fitness evaluation in each generation. Use 0 to use the entire dataset. */
    SelectionType selection_type;   /**< The method for selecting the fittest individuals for reproduction (e.g., `TOURNAMENT_SELECTION`). */
    int tournament_size;            /**< The number of individuals to compete in a tournament, if `TOURNAMENT_SELECTION` is used. */
    int elitism_count;              /**< The number of top-performing individuals to carry over to the next generation without modification. */
    ActivationType activation_hidden; /**< The activation function to use for all hidden layers (e.g., `RELU`, `SIGMOID`). */
    ActivationType activation_output; /**< The activation function to use for the output layer (e.g., `SIGMOID`, `LINEAR`). */
    CrossoverType crossover_type;   /**< The crossover strategy to use for combining parent networks (e.g., `UNIFORM_CROSSOVER`). */
    MutationType mutation_type;     /**< The mutation strategy to use (e.g., `GAUSSIAN_MUTATION`, `RANDOM_MUTATION`). */
    double mutation_std_dev;        /**< The standard deviation for Gaussian mutation. Only used if `mutation_type` is `GAUSSIAN_MUTATION`. */
    bool logging;                   /**< If true, prints progress information (generation number, fitness scores) to the console during training. */
    int early_stopping_patience;    /**< Number of generations with no improvement in validation accuracy to wait before stopping training. Set to 0 to disable. */
    double early_stopping_threshold;/**< The minimum improvement in validation accuracy required to reset the patience counter for early stopping. */
} GannTrainParams;

/**
 * @brief Creates a `GannTrainParams` struct with sensible default values.
 * @details This function provides a convenient starting point for training.
 * The user is still required to set the `architecture` and `num_layers` fields
 * manually, as these are specific to the problem being solved.
 * @return A `GannTrainParams` struct populated with default values.
 */
GannTrainParams gann_create_default_params(void);


// --- Function Pointer Typedefs for Extensibility ---
typedef NetworkFitness* (*SelectionFunction)(NetworkFitness*, int, int*, SelectionType, int);
typedef NeuralNetwork* (*CrossoverFunction)(const NeuralNetwork*, const NeuralNetwork*, CrossoverType);
typedef void (*MutationFunction)(NeuralNetwork*, float, float, MutationType, double, int, int, double);


/**
 * @brief A struct for the `gann_evolve` function, allowing for custom genetic operators.
 * @details This struct is used to pass a combination of base training parameters and
 * function pointers for custom selection, crossover, and mutation logic to the
 * `gann_evolve` function.
 */
typedef struct {
    GannTrainParams base_params;       /**< The base training parameters. */
    SelectionFunction selection_func;  /**< A function pointer to the selection operator. */
    CrossoverFunction crossover_func;  /**< A function pointer to the crossover operator. */
    MutationFunction mutation_func;    /**< A function pointer to the mutation operator. */
} GannEvolveParams;


/**
 * @brief Evolves a population of neural networks using custom genetic operators.
 * @details This is an advanced version of `gann_train` that offers greater
 * flexibility by allowing the user to provide their own implementations for the
 * core genetic operators: selection, crossover, and mutation.
 * @param params The evolution parameters, including the base parameters and function pointers to the genetic operators.
 * @param train_dataset The dataset to train the network on.
 * @param validation_dataset An optional dataset for validation, used for early stopping. Can be `NULL`.
 * @return A pointer to the best-trained `NeuralNetwork`. The caller is responsible for freeing this network using `nn_free()`.
 * @return `NULL` on failure. If `NULL` is returned, call `gann_get_last_error()` to get the specific error code.
 */
NeuralNetwork* gann_evolve(const GannEvolveParams* params, const Dataset* train_dataset, const Dataset* validation_dataset);


/**
 * @brief Trains a new neural network using a genetic algorithm with default operators.
 * @details This function encapsulates the entire genetic algorithm training loop,
 * including population initialization, evaluation, selection, crossover, and mutation.
 * It uses the standard genetic operators built into the library.
 * @param params The training parameters, configured in a `GannTrainParams` struct.
 * @param train_dataset The dataset to train the network on.
 * @param validation_dataset An optional dataset for validation, used for early stopping. Can be `NULL`.
 * @return A pointer to the best-trained `NeuralNetwork`. The caller is responsible for freeing this network using `nn_free()`.
 * @return `NULL` on failure. If `NULL` is returned, call `gann_get_last_error()` to get the specific error code.
 */
NeuralNetwork* gann_train(const GannTrainParams* params, const Dataset* train_dataset, const Dataset* validation_dataset);



/**
 * @brief Trains a new neural network using backpropagation.
 * @details This function trains a single neural network using the backpropagation
 * algorithm with a chosen optimizer (e.g., SGD, Adam, RMSprop).
 * @param params The backpropagation training parameters, configured in a `GannBackpropParams` struct.
 * @param train_dataset The dataset to train the network on.
 * @param validation_dataset An optional dataset for validation, used for tracking performance during training. Can be `NULL`.
 * @return A pointer to the trained `NeuralNetwork`. The caller is responsible for freeing this network using `nn_free()`.
 * @return `NULL` on failure. If `NULL` is returned, call `gann_get_last_error()` to get the specific error code.
 */
NeuralNetwork* gann_train_with_backprop(const GannBackpropParams* params, const Dataset* train_dataset, const Dataset* validation_dataset);


/**
 * @brief Makes a prediction on a single input vector using a trained network.
 * @details This function performs a forward pass through the network with the
 * given input data and returns the index of the output neuron with the highest activation.
 * @param net The trained neural network.
 * @param input A flat array of input data (e.g., pixel values). The size of this array must match the network's input layer size.
 * @return The index of the predicted class (e.g., the digit 0-9).
 * @return -1 on failure. If -1 is returned, call `gann_get_last_error()` to get the specific error code.
 */
int gann_predict(const NeuralNetwork* net, const double* input);

/**
 * @brief Evaluates the network's accuracy on a given dataset.
 * @details This function iterates through the entire dataset, makes a prediction
 * for each item, and calculates the overall accuracy as the ratio of correct
 * predictions to the total number of items.
 * @param net The trained neural network.
 * @param dataset The dataset to evaluate on (e.g., a test set or validation set).
 * @return The accuracy of the network on the dataset, as a value from 0.0 to 1.0.
 * @return On failure, returns 0.0 and sets an error code. Call `gann_get_last_error()` to check for errors.
 */
double gann_evaluate(const NeuralNetwork* net, const Dataset* dataset);

#endif // GANN_H
