#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include "matrix.h"

// --- Struct Definitions ---

/**
 * @brief Enumeration of supported activation functions.
 */
typedef enum {
    SIGMOID,    /**< Sigmoid activation function. Maps input to a range between 0 and 1. */
    RELU,       /**< Rectified Linear Unit (ReLU) activation function. Returns `max(0, x)`. */
    LEAKY_RELU  /**< Leaky ReLU activation function. A variant of ReLU that allows a small, non-zero gradient when the unit is not active. */
} ActivationType;

/**
 * @brief Represents the state for optimizers like Adam and RMSprop.
 * @details This struct holds the moving averages of the gradients required by
 * certain optimization algorithms. It is allocated internally when needed,
 * for example, during backpropagation training.
 */
typedef struct OptimizerState {
    Matrix** m_weights; /**< First moment (mean) of the gradients for weights. */
    Matrix** v_weights; /**< Second moment (uncentered variance) of the gradients for weights. */
    Matrix** m_biases;  /**< First moment (mean) of the gradients for biases. */
    Matrix** v_biases;  /**< Second moment (uncentered variance) of the gradients for biases. */
} OptimizerState;

/**
 * @brief Represents a feedforward neural network.
 * @details This struct contains the entire state of a neural network, including its
 * architecture, weights, biases, and activation functions. It also holds an
 * optional pointer to an optimizer state for training with backpropagation.
 */
typedef struct {
    int num_layers;                   /**< The total number of layers in the network (input + hidden + output). */
    int* architecture;                /**< An array of integers defining the number of neurons in each layer, e.g., `{784, 128, 10}`. */
    Matrix** weights;                 /**< An array of weight matrices. `weights[i]` connects layer `i` and `i+1`. */
    Matrix** biases;                  /**< An array of bias matrices (represented as row vectors). `biases[i]` is for layer `i+1`. */
    ActivationType activation_hidden; /**< The activation function used for all hidden layers. */
    ActivationType activation_output; /**< The activation function used for the output layer. */
    OptimizerState* optimizer_state;  /**< A pointer to the optimizer state, used only for backpropagation training. `NULL` otherwise. */
} NeuralNetwork;

/**
 * @brief Applies an activation function element-wise to a matrix, modifying it in place.
 * @param m The matrix to modify.
 * @param activation_type The type of activation function to apply (e.g., `SIGMOID`, `RELU`).
 */
void nn_apply_activation(Matrix* m, ActivationType activation_type);

/**
 * @brief Applies the derivative of an activation function element-wise to a matrix, modifying it in place.
 * @details This is a crucial step in the backpropagation algorithm, used to calculate gradients.
 * The input matrix should contain the values *before* the activation function was applied (the weighted sum).
 * @param m The matrix of pre-activation values to which the derivative will be applied.
 * @param activation_type The type of activation function derivative to apply.
 */
void nn_apply_activation_derivative(Matrix* m, ActivationType activation_type);


// --- Neural Network Operations ---

/**
 * @brief Creates and allocates a new neural network structure.
 * @details This function allocates memory for a `NeuralNetwork` struct and all its
 * internal components (weights, biases, architecture array). The weights and biases
 * are not initialized with meaningful values; call `nn_init()` to initialize them.
 * The caller is responsible for freeing the network using `nn_free()`.
 * @param num_layers The total number of layers (input, hidden, and output).
 * @param architecture An array of integers specifying the number of neurons in each layer. A deep copy of this array is made.
 * @param activation_hidden The activation function to be used for the hidden layers.
 * @param activation_output The activation function to be used for the output layer.
 * @return A pointer to the newly created `NeuralNetwork`, or `NULL` if allocation fails.
 */
NeuralNetwork* nn_create(int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output);

/**
 * @brief Initializes the optimizer state for a neural network.
 * @details This function allocates memory for the `OptimizerState` struct and its
 * internal matrices. This is only necessary when training the network using backpropagation
 * with optimizers like Adam or RMSprop that require state.
 * @param net The neural network for which to initialize the optimizer state.
 * @return 1 on success, 0 on failure (e.g., memory allocation failed).
 */
int nn_init_optimizer_state(NeuralNetwork* net);

/**
 * @brief Frees all memory allocated for a neural network.
 * @details This function deallocates the network struct itself, its weights, biases,
 * architecture array, and its optimizer state if it has been allocated.
 * @param net The neural network to free. It's safe to pass `NULL`.
 */
void nn_free(NeuralNetwork* net);

/**
 * @brief Initializes the weights of a neural network with random values.
 * @details It uses a variant of Xavier/Glorot initialization to set the initial
 * weights, which helps prevent gradients from vanishing or exploding during the
 * initial stages of training. Biases are initialized to zero.
 * @param net The neural network to initialize.
 */
void nn_init(NeuralNetwork* net);

/**
 * @brief Performs a forward pass through the network to compute an output.
 * @param net The neural network.
 * @param input The input matrix, with dimensions `1 x num_input_neurons`.
 * @return A new matrix containing the output of the network, with dimensions `1 x num_output_neurons`.
 * The caller is responsible for freeing this matrix using `free_matrix()`.
 * @return `NULL` on failure (e.g., invalid input dimensions).
 */
Matrix* nn_forward_pass(const NeuralNetwork* net, const Matrix* input);

/**
 * @brief Creates a deep copy of a neural network.
 * @details This function creates a new, independent copy of the source network,
 * including its architecture, weights, biases, and optimizer state.
 * @param src_net The source network to clone.
 * @return A pointer to the newly cloned `NeuralNetwork`. The caller is responsible for
 * freeing this network using `nn_free()`. Returns `NULL` on failure.
 */
NeuralNetwork* nn_clone(const NeuralNetwork* src_net);

/**
 * @brief Saves a neural network's structure and parameters to a binary file.
 * @param net The neural network to save.
 * @param filepath The path to the file where the network will be saved.
 * @return 1 on success, 0 on failure (e.g., file could not be opened).
 */
int nn_save(const NeuralNetwork* net, const char* filepath);

/**
 * @brief Loads a neural network from a binary file.
 * @details This function reconstructs a neural network that was previously saved
 * using `nn_save()`.
 * @param filepath The path to the file to load.
 * @return A pointer to the loaded `NeuralNetwork`. The caller is responsible for freeing
 * this network using `nn_free()`. Returns `NULL` on failure (e.g., file not found, format error).
 */
NeuralNetwork* nn_load(const char* filepath);

#endif // NEURAL_NETWORK_H
