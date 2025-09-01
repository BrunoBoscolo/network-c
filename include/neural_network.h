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
} NeuralNetwork;

/**
 * @brief Applies an activation function element-wise to a matrix.
 * @param m The matrix to modify.
 * @param activation_type The type of activation function to apply.
 */
void apply_activation(Matrix* m, ActivationType activation_type);

// --- Neural Network Operations ---

/**
 * @brief Creates a new neural network.
 * @param num_layers The number of layers.
 * @param architecture An array of integers specifying the number of neurons in each layer.
 * @param activation_hidden The activation function for the hidden layers.
 * @param activation_output The activation function for the output layer.
 * @return A pointer to the newly created NeuralNetwork.
 */
NeuralNetwork* create_neural_network(int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output);

/**
 * @brief Frees the memory allocated for a neural network.
 * @param net The neural network to free.
 */
void free_neural_network(NeuralNetwork* net);

/**
 * @brief Initializes the weights and biases of a neural network with random values.
 * @param net The neural network to initialize.
 */
void initialize_network(NeuralNetwork* net);

/**
 * @brief Performs a forward pass through the network.
 * @param net The neural network.
 * @param input The input matrix.
 * @return A new matrix containing the output of the network.
 */
Matrix* forward_pass(const NeuralNetwork* net, const Matrix* input);

/**
 * @brief Mutates the weights and biases of a neural network.
 * @param net The neural network to mutate.
 * @param mutation_rate The magnitude of the mutation.
 * @param mutation_chance The chance of a mutation occurring.
 */
void mutate_network(NeuralNetwork* net, float mutation_rate, float mutation_chance);

/**
 * @brief Creates a deep copy of a neural network.
 * @param src_net The source network to clone.
 * @return A pointer to the newly cloned NeuralNetwork.
 */
NeuralNetwork* clone_network(const NeuralNetwork* src_net);

/**
 * @brief Saves a neural network to a file.
 * @param net The neural network to save.
 * @param filepath The path to the file.
 * @return 1 on success, 0 on failure.
 */
int save_network(const NeuralNetwork* net, const char* filepath);

/**
 * @brief Loads a neural network from a file.
 * @param filepath The path to the file.
 * @return A pointer to the loaded NeuralNetwork.
 */
NeuralNetwork* load_network(const char* filepath);

#endif // NEURAL_NETWORK_H
