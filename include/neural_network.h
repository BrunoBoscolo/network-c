#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>

// --- Struct Definitions ---

// Represents a 2D matrix
typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

// Represents a feedforward neural network
typedef struct {
    int num_layers;
    int* architecture; // Array of layer sizes, e.g., [3, 5, 2]
    Matrix** weights;   // Array of weight matrices
    Matrix** biases;    // Array of bias matrices (vectors)
} NeuralNetwork;

// --- Matrix Operations ---

Matrix* create_matrix(int rows, int cols);
void free_matrix(Matrix* m);
void print_matrix(const Matrix* m);
Matrix* dot_product(const Matrix* m1, const Matrix* m2);
void add_bias(Matrix* m, const Matrix* bias);
void apply_sigmoid(Matrix* m);

// --- Neural Network Operations ---

NeuralNetwork* create_neural_network(int num_layers, const int* architecture);
void free_neural_network(NeuralNetwork* net);
void initialize_network(NeuralNetwork* net);
Matrix* forward_pass(const NeuralNetwork* net, const Matrix* input);
void mutate_network(NeuralNetwork* net, float mutation_rate, float mutation_chance);
NeuralNetwork* clone_network(const NeuralNetwork* src_net);

int save_network(const NeuralNetwork* net, const char* filepath);
NeuralNetwork* load_network(const char* filepath);

#endif // NEURAL_NETWORK_H
