#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include "matrix.h"

// --- Struct Definitions ---

// Enum for activation functions
typedef enum {
    SIGMOID,
    RELU,
    LEAKY_RELU
} ActivationType;

// Represents a feedforward neural network
typedef struct {
    int num_layers;
    int* architecture; // Array of layer sizes, e.g., [3, 5, 2]
    Matrix** weights;   // Array of weight matrices
    Matrix** biases;    // Array of bias matrices (vectors)
    ActivationType activation_hidden; // Activation for hidden layers
    ActivationType activation_output; // Activation for output layer

    // Optimizer state - only allocated for backprop training
    struct OptimizerState* optimizer_state;
} NeuralNetwork;

// Represents the state for optimizers like Adam and RMSprop
typedef struct OptimizerState {
    Matrix** m_weights; // First moment for weights
    Matrix** v_weights; // Second moment for weights
    Matrix** m_biases;  // First moment for biases
    Matrix** v_biases;  // Second moment for biases
} OptimizerState;

/**
 * @brief Applies an activation function element-wise to a matrix.
 * @param m The matrix to modify.
 * @param activation_type The type of activation function to apply.
 */
void nn_apply_activation(Matrix* m, ActivationType activation_type);

/**
 * @brief Applies the derivative of an activation function element-wise to a matrix.
 * This is used during backpropagation.
 * @param m The matrix to which the derivative will be applied.
 * @param activation_type The type of activation function derivative to apply.
 */
void nn_apply_activation_derivative(Matrix* m, ActivationType activation_type);


// --- Neural Network Operations ---

/**
 * @brief Creates a new neural network. The caller is responsible for freeing the network
 * using `nn_free()`. The network's weights and biases are not initialized.
 * @param num_layers The number of layers.
 * @param architecture An array of integers specifying the number of neurons in each layer. The library makes a copy of this array.
 * @param activation_hidden The activation function for the hidden layers.
 * @param activation_output The activation function for the output layer.
 * @return A pointer to the newly created NeuralNetwork, or NULL on failure.
 */
NeuralNetwork* nn_create(int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output);

/**
 * @brief Initializes the optimizer state for a neural network. This is only needed
 * for training with backpropagation.
 * @param net The neural network for which to initialize the optimizer state.
 * @return 1 on success, 0 on failure.
 */
int nn_init_optimizer_state(NeuralNetwork* net);

/**
 * @brief Frees all memory allocated for a neural network, including its weights and biases.
 * @param net The neural network to free.
 */
void nn_free(NeuralNetwork* net);

/**
 * @brief Initializes the weights and biases of a neural network with random values
 * using a variant of Xavier/Glorot initialization.
 * @param net The neural network to initialize.
 */
void nn_init(NeuralNetwork* net);

/**
 * @brief Performs a forward pass through the network.
 * @param net The neural network.
 * @param input The input matrix.
 * @return A new matrix containing the output of the network. The caller is responsible
 * for freeing this matrix using `free_matrix()`. Returns NULL on failure.
 */
Matrix* nn_forward_pass(const NeuralNetwork* net, const Matrix* input);

/**
 * @brief Creates a deep copy of a neural network.
 * @param src_net The source network to clone.
 * @return A pointer to the newly cloned NeuralNetwork. The caller is responsible for
 * freeing this network using `nn_free()`. Returns NULL on failure.
 */
NeuralNetwork* nn_clone(const NeuralNetwork* src_net);

/**
 * @brief Saves a neural network to a file.
 * @param net The neural network to save.
 * @param filepath The path to the file.
 * @return 1 on success, 0 on failure.
 */
int nn_save(const NeuralNetwork* net, const char* filepath);

/**
 * @brief Loads a neural network from a file.
 * @param filepath The path to the file.
 * @return A pointer to the loaded NeuralNetwork. The caller is responsible for freeing
 * this network using `nn_free()`. Returns NULL on failure.
 */
NeuralNetwork* nn_load(const char* filepath);

#endif // NEURAL_NETWORK_H
