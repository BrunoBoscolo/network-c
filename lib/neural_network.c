#include "neural_network.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- Activation Functions ---

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// ReLU activation function
double relu(double x) {
    return x > 0 ? x : 0;
}

// Leaky ReLU activation function
double leaky_relu(double x) {
    return x > 0 ? x : 0.01 * x;
}

// --- Activation Function Derivatives ---

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// Derivative of the ReLU function
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Derivative of the Leaky ReLU function
double leaky_relu_derivative(double x) {
    return x > 0 ? 1 : 0.01;
}


// Applies the specified activation function element-wise to a matrix
void apply_activation(Matrix* m, ActivationType activation_type) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double* val = &m->data[i][j];
            switch (activation_type) {
                case SIGMOID:
                    *val = sigmoid(*val);
                    break;
                case RELU:
                    *val = relu(*val);
                    break;
                case LEAKY_RELU:
                    *val = leaky_relu(*val);
                    break;
            }
        }
    }
}

// Applies the derivative of the specified activation function element-wise to a matrix
void apply_activation_derivative(Matrix* m, ActivationType activation_type) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double* val = &m->data[i][j];
            switch (activation_type) {
                case SIGMOID:
                    *val = sigmoid_derivative(*val);
                    break;
                case RELU:
                    *val = relu_derivative(*val);
                    break;
                case LEAKY_RELU:
                    *val = leaky_relu_derivative(*val);
                    break;
            }
        }
    }
}


// --- Neural Network Operations Implementation ---

// Creates and allocates memory for a neural network
NeuralNetwork* create_neural_network(int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output) {
    if (num_layers < 2) return NULL; // A network must have at least an input and an output layer

    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) return NULL;

    net->num_layers = num_layers;
    net->activation_hidden = activation_hidden;
    net->activation_output = activation_output;
    net->architecture = (int*)malloc(num_layers * sizeof(int));
    if (!net->architecture) {
        free(net);
        return NULL;
    }
    for(int i=0; i<num_layers; i++) net->architecture[i] = architecture[i];

    net->weights = (Matrix**)malloc((num_layers - 1) * sizeof(Matrix*));
    if (!net->weights) {
        free(net->architecture);
        free(net);
        return NULL;
    }

    net->biases = (Matrix**)malloc((num_layers - 1) * sizeof(Matrix*));
    if (!net->biases) {
        free(net->weights);
        free(net->architecture);
        free(net);
        return NULL;
    }

    for (int i = 0; i < num_layers - 1; i++) {
        net->weights[i] = create_matrix(architecture[i], architecture[i+1]);
        if (!net->weights[i]) {
            // Rollback
            for (int j = 0; j < i; j++) free_matrix(net->weights[j]);
            free(net->weights);
            free(net->biases); // Biases for this layer were not allocated yet
            free(net->architecture);
            free(net);
            return NULL;
        }
        net->biases[i] = create_matrix(1, architecture[i+1]);
        if (!net->biases[i]) {
            // Rollback
            free_matrix(net->weights[i]); // Free the weight matrix for the current layer
            for (int j = 0; j < i; j++) {
                free_matrix(net->weights[j]);
                free_matrix(net->biases[j]);
            }
            free(net->weights);
            free(net->biases);
            free(net->architecture);
            free(net);
            return NULL;
        }
    }

    return net;
}

// Initializes network with random weights and zero biases
void initialize_network(NeuralNetwork* net) {
    for (int i = 0; i < net->num_layers - 1; i++) {
        // He-et-al initialization for weights
        for (int r = 0; r < net->weights[i]->rows; r++) {
            for (int c = 0; c < net->weights[i]->cols; c++) {
                net->weights[i]->data[r][c] = ((double)rand() / RAND_MAX) * sqrt(2.0 / net->architecture[i]);
            }
        }
        // Biases are already initialized to zero by calloc in create_matrix
    }
}

// Frees all memory associated with a neural network
void free_neural_network(NeuralNetwork* net) {
    if (!net) return;
    for (int i = 0; i < net->num_layers - 1; i++) {
        free_matrix(net->weights[i]);
        free_matrix(net->biases[i]);
    }
    free(net->weights);
    free(net->biases);
    free(net->architecture);
    free(net);
}

// Performs a forward pass through the network
Matrix* forward_pass(const NeuralNetwork* net, const Matrix* input) {
    if (input->cols != net->architecture[0]) return NULL;

    Matrix* current_output = (Matrix*)input; // Start with the input, no copy
    int input_was_copied = 0; // Flag to track if we need to free current_output

    for (int i = 0; i < net->num_layers - 1; i++) {
        Matrix* next_output = dot_product(current_output, net->weights[i]);
        if (!next_output) {
            if (input_was_copied) free_matrix(current_output);
            return NULL;
        }

        add_bias(next_output, net->biases[i]);

        // Apply activation function
        // Use hidden layer activation for all but the last layer
        if (i < net->num_layers - 2) {
            apply_activation(next_output, net->activation_hidden);
        } else {
            apply_activation(next_output, net->activation_output);
        }

        if (input_was_copied) {
            free_matrix(current_output);
        }

        current_output = next_output;
        input_was_copied = 1; // From now on, current_output is a new matrix
    }

    return current_output;
}

// Function to generate a random number from a Gaussian distribution (Box-Muller transform)
double randn(double mu, double sigma) {
    double u1, u2, w, mult;
    static double x1, x2;
    static int call = 0;

    if (call == 1) {
        call = !call;
        return (mu + sigma * (double)x2);
    }

    do {
        u1 = -1 + ((double)rand() / RAND_MAX) * 2;
        u2 = -1 + ((double)rand() / RAND_MAX) * 2;
        w = u1 * u1 + u2 * u2;
    } while (w >= 1 || w == 0);

    mult = sqrt((-2 * log(w)) / w);
    x1 = u1 * mult;
    x2 = u2 * mult;

    call = !call;

    return (mu + sigma * (double)x1);
}

// Applies random uniform mutation
void random_uniform_mutation(NeuralNetwork* net, float mutation_rate, float mutation_chance) {
    // Mutate weights
    for (int i = 0; i < net->num_layers - 1; i++) {
        for (int r = 0; r < net->weights[i]->rows; r++) {
            for (int c = 0; c < net->weights[i]->cols; c++) {
                if (((double)rand() / RAND_MAX) < mutation_chance) {
                    net->weights[i]->data[r][c] += ((double)rand() / RAND_MAX - 0.5) * mutation_rate;
                }
            }
        }
    }
    // Mutate biases
    for (int i = 0; i < net->num_layers - 1; i++) {
        for (int c = 0; c < net->biases[i]->cols; c++) {
             if (((double)rand() / RAND_MAX) < mutation_chance) {
                net->biases[i]->data[0][c] += ((double)rand() / RAND_MAX - 0.5) * mutation_rate;
            }
        }
    }
}


// Applies Gaussian mutation
void gaussian_mutation(NeuralNetwork* net, float mutation_chance, double mutation_std_dev) {
    // Mutate weights
    for (int i = 0; i < net->num_layers - 1; i++) {
        for (int r = 0; r < net->weights[i]->rows; r++) {
            for (int c = 0; c < net->weights[i]->cols; c++) {
                if (((double)rand() / RAND_MAX) < mutation_chance) {
                    net->weights[i]->data[r][c] += randn(0, mutation_std_dev);
                }
            }
        }
    }
    // Mutate biases
    for (int i = 0; i < net->num_layers - 1; i++) {
        for (int c = 0; c < net->biases[i]->cols; c++) {
             if (((double)rand() / RAND_MAX) < mutation_chance) {
                net->biases[i]->data[0][c] += randn(0, mutation_std_dev);
            }
        }
    }
}

// Mutates the network's parameters
void mutate_network(NeuralNetwork* net, float mutation_rate, float mutation_chance, MutationType mutation_type, double mutation_std_dev) {
    switch (mutation_type) {
        case RANDOM_UNIFORM:
            random_uniform_mutation(net, mutation_rate, mutation_chance);
            break;
        case GAUSSIAN:
            gaussian_mutation(net, mutation_chance, mutation_std_dev);
            break;
        default:
            random_uniform_mutation(net, mutation_rate, mutation_chance);
            break;
    }
}

// Creates a deep copy of a neural network
NeuralNetwork* clone_network(const NeuralNetwork* src_net) {
    if (!src_net) return NULL;

    NeuralNetwork* new_net = create_neural_network(src_net->num_layers, src_net->architecture, src_net->activation_hidden, src_net->activation_output);
    if (!new_net) return NULL;

    for (int i = 0; i < src_net->num_layers - 1; i++) {
        for (int r = 0; r < src_net->weights[i]->rows; r++) {
            for (int c = 0; c < src_net->weights[i]->cols; c++) {
                new_net->weights[i]->data[r][c] = src_net->weights[i]->data[r][c];
            }
        }
        for (int c = 0; c < src_net->biases[i]->cols; c++) {
            new_net->biases[i]->data[0][c] = src_net->biases[i]->data[0][c];
        }
    }

    return new_net;
}

int save_network(const NeuralNetwork* net, const char* filepath) {
    FILE* file = fopen(filepath, "w");
    if (!file) {
        perror("Failed to open file for writing");
        return 0; // Failure
    }

    // Write architecture
    fprintf(file, "%d\n", net->num_layers);
    for (int i = 0; i < net->num_layers; i++) {
        fprintf(file, "%d ", net->architecture[i]);
    }
    fprintf(file, "\n");

    // Write weights and biases
    for (int i = 0; i < net->num_layers - 1; i++) {
        // Weights
        for (int r = 0; r < net->weights[i]->rows; r++) {
            for (int c = 0; c < net->weights[i]->cols; c++) {
                fprintf(file, "%.17g ", net->weights[i]->data[r][c]);
            }
            fprintf(file, "\n");
        }
        // Biases
        for (int c = 0; c < net->biases[i]->cols; c++) {
            fprintf(file, "%.17g ", net->biases[i]->data[0][c]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    return 1; // Success
}

NeuralNetwork* load_network(const char* filepath) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        perror("Failed to open file for reading");
        return NULL;
    }

    // Read architecture
    int num_layers;
    if (fscanf(file, "%d", &num_layers) != 1) {
        fclose(file);
        return NULL;
    }

    int* architecture = (int*)malloc(num_layers * sizeof(int));
    if (!architecture) {
        fclose(file);
        return NULL;
    }
    for (int i = 0; i < num_layers; i++) {
        if (fscanf(file, "%d", &architecture[i]) != 1) {
            free(architecture);
            fclose(file);
            return NULL; // Failed to read architecture
        }
    }

    // For now, hardcode activation functions when loading older formats.
    // A future improvement would be to version the file format.
    NeuralNetwork* net = create_neural_network(num_layers, architecture, SIGMOID, SIGMOID);
    free(architecture); // create_neural_network makes a copy
    if (!net) {
        fclose(file);
        return NULL;
    }

    // Read weights and biases
    for (int i = 0; i < net->num_layers - 1; i++) {
        // Weights
        for (int r = 0; r < net->weights[i]->rows; r++) {
            for (int c = 0; c < net->weights[i]->cols; c++) {
                if (fscanf(file, "%lf", &net->weights[i]->data[r][c]) != 1) {
                    free_neural_network(net);
                    fclose(file);
                    return NULL; // Failed to read weight
                }
            }
        }
        // Biases
        for (int c = 0; c < net->biases[i]->cols; c++) {
            if (fscanf(file, "%lf", &net->biases[i]->data[0][c]) != 1) {
                free_neural_network(net);
                fclose(file);
                return NULL; // Failed to read bias
            }
        }
    }

    fclose(file);
    return net;
}
