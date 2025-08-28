#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- Matrix Operations Implementation ---

// Creates and allocates memory for a new matrix
Matrix* create_matrix(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;

    m->rows = rows;
    m->cols = cols;
    m->data = (double**)malloc(rows * sizeof(double*));
    if (!m->data) {
        free(m);
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        m->data[i] = (double*)calloc(cols, sizeof(double));
        if (!m->data[i]) {
            // Rollback allocation on failure
            for (int j = 0; j < i; j++) free(m->data[j]);
            free(m->data);
            free(m);
            return NULL;
        }
    }
    return m;
}

// Frees the memory of a matrix
void free_matrix(Matrix* m) {
    if (!m) return;
    for (int i = 0; i < m->rows; i++) {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

// Prints the matrix data (for debugging)
void print_matrix(const Matrix* m) {
    if (!m) return;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%f ", m->data[i][j]);
        }
        printf("\n");
    }
}

// Computes the dot product of two matrices
Matrix* dot_product(const Matrix* m1, const Matrix* m2) {
    if (m1->cols != m2->rows) return NULL;

    Matrix* result = create_matrix(m1->rows, m2->cols);
    if (!result) return NULL;

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m2->cols; j++) {
            for (int k = 0; k < m1->cols; k++) {
                result->data[i][j] += m1->data[i][k] * m2->data[k][j];
            }
        }
    }
    return result;
}

// Adds a bias vector to each row of a matrix
void add_bias(Matrix* m, const Matrix* bias) {
    if (m->cols != bias->cols || bias->rows != 1) return;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] += bias->data[0][j];
        }
    }
}

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Applies the sigmoid function element-wise to a matrix
void apply_sigmoid(Matrix* m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] = sigmoid(m->data[i][j]);
        }
    }
}

// --- Neural Network Operations Implementation ---

// Creates and allocates memory for a neural network
NeuralNetwork* create_neural_network(int num_layers, const int* architecture) {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->num_layers = num_layers;
    net->architecture = (int*)malloc(num_layers * sizeof(int));
    for(int i=0; i<num_layers; i++) net->architecture[i] = architecture[i];

    net->weights = (Matrix**)malloc((num_layers - 1) * sizeof(Matrix*));
    net->biases = (Matrix**)malloc((num_layers - 1) * sizeof(Matrix*));

    for (int i = 0; i < num_layers - 1; i++) {
        net->weights[i] = create_matrix(architecture[i], architecture[i+1]);
        net->biases[i] = create_matrix(1, architecture[i+1]);
    }

    initialize_network(net);

    return net;
}

// Initializes network with random weights and zero biases
void initialize_network(NeuralNetwork* net) {
    // Seed random number generator
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }

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

    Matrix* current_output = create_matrix(input->rows, input->cols);
    for(int i=0; i<input->rows; i++) {
        for(int j=0; j<input->cols; j++) {
            current_output->data[i][j] = input->data[i][j];
        }
    }


    for (int i = 0; i < net->num_layers - 1; i++) {
        Matrix* next_output = dot_product(current_output, net->weights[i]);
        add_bias(next_output, net->biases[i]);

        apply_sigmoid(next_output);

        free_matrix(current_output);
        current_output = next_output;
    }

    return current_output;
}

// Mutates the network's parameters
void mutate_network(NeuralNetwork* net, float mutation_rate, float mutation_chance) {
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

// Creates a deep copy of a neural network
NeuralNetwork* clone_network(const NeuralNetwork* src_net) {
    if (!src_net) return NULL;

    NeuralNetwork* new_net = create_neural_network(src_net->num_layers, src_net->architecture);

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

    NeuralNetwork* net = create_neural_network(num_layers, architecture);
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
