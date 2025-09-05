#include "crossover.h"
#include <stdlib.h>
#include <stdio.h>

// Performs uniform crossover between two parent networks.
// For each weight and bias, the child's value is randomly taken from one of the two parents.
static NeuralNetwork* uniform_crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2) {
    if (!parent1 || !parent2 || parent1->num_layers != parent2->num_layers) {
        return NULL;
    }

    // Create a new network with the same architecture
    NeuralNetwork* child = nn_create(parent1->num_layers, parent1->architecture, parent1->activation_hidden, parent1->activation_output);
    if (!child) return NULL;

    // Perform uniform crossover for weights and biases
    for (int i = 0; i < parent1->num_layers - 1; i++) {
        // Weights
        for (int r = 0; r < parent1->weights[i]->rows; r++) {
            for (int c = 0; c < parent1->weights[i]->cols; c++) {
                if ((double)rand() / RAND_MAX > 0.5) {
                    child->weights[i]->data[r][c] = parent1->weights[i]->data[r][c];
                } else {
                    child->weights[i]->data[r][c] = parent2->weights[i]->data[r][c];
                }
            }
        }
        // Biases
        for (int c = 0; c < parent1->biases[i]->cols; c++) {
            if ((double)rand() / RAND_MAX > 0.5) {
                child->biases[i]->data[0][c] = parent1->biases[i]->data[0][c];
            } else {
                child->biases[i]->data[0][c] = parent2->biases[i]->data[0][c];
            }
        }
    }

    return child;
}

// Performs single-point crossover between two parent networks.
static NeuralNetwork* single_point_crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2) {
    if (!parent1 || !parent2 || parent1->num_layers != parent2->num_layers) {
        return NULL;
    }

    NeuralNetwork* child = nn_create(parent1->num_layers, parent1->architecture, parent1->activation_hidden, parent1->activation_output);
    if (!child) return NULL;

    int total_weights = 0;
    for (int i = 0; i < parent1->num_layers - 1; i++) {
        total_weights += parent1->weights[i]->rows * parent1->weights[i]->cols;
        total_weights += parent1->biases[i]->cols;
    }

    int crossover_point = rand() % total_weights;
    int current_weight = 0;

    for (int i = 0; i < parent1->num_layers - 1; i++) {
        // Weights
        for (int r = 0; r < parent1->weights[i]->rows; r++) {
            for (int c = 0; c < parent1->weights[i]->cols; c++) {
                if (current_weight < crossover_point) {
                    child->weights[i]->data[r][c] = parent1->weights[i]->data[r][c];
                } else {
                    child->weights[i]->data[r][c] = parent2->weights[i]->data[r][c];
                }
                current_weight++;
            }
        }
        // Biases
        for (int c = 0; c < parent1->biases[i]->cols; c++) {
            if (current_weight < crossover_point) {
                child->biases[i]->data[0][c] = parent1->biases[i]->data[0][c];
            } else {
                child->biases[i]->data[0][c] = parent2->biases[i]->data[0][c];
            }
            current_weight++;
        }
    }

    return child;
}

// Performs two-point crossover between two parent networks.
static NeuralNetwork* two_point_crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2) {
    if (!parent1 || !parent2 || parent1->num_layers != parent2->num_layers) {
        return NULL;
    }

    NeuralNetwork* child = nn_create(parent1->num_layers, parent1->architecture, parent1->activation_hidden, parent1->activation_output);
    if (!child) return NULL;

    int total_weights = 0;
    for (int i = 0; i < parent1->num_layers - 1; i++) {
        total_weights += parent1->weights[i]->rows * parent1->weights[i]->cols;
        total_weights += parent1->biases[i]->cols;
    }

    int crossover_point1 = rand() % total_weights;
    int crossover_point2 = rand() % total_weights;
    if (crossover_point1 > crossover_point2) {
        int temp = crossover_point1;
        crossover_point1 = crossover_point2;
        crossover_point2 = temp;
    }

    int current_weight = 0;

    for (int i = 0; i < parent1->num_layers - 1; i++) {
        // Weights
        for (int r = 0; r < parent1->weights[i]->rows; r++) {
            for (int c = 0; c < parent1->weights[i]->cols; c++) {
                if (current_weight >= crossover_point1 && current_weight < crossover_point2) {
                    child->weights[i]->data[r][c] = parent2->weights[i]->data[r][c];
                } else {
                    child->weights[i]->data[r][c] = parent1->weights[i]->data[r][c];
                }
                current_weight++;
            }
        }
        // Biases
        for (int c = 0; c < parent1->biases[i]->cols; c++) {
            if (current_weight >= crossover_point1 && current_weight < crossover_point2) {
                child->biases[i]->data[0][c] = parent2->biases[i]->data[0][c];
            } else {
                child->biases[i]->data[0][c] = parent1->biases[i]->data[0][c];
            }
            current_weight++;
        }
    }

    return child;
}

// Performs arithmetic crossover between two parent networks.
static NeuralNetwork* arithmetic_crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2) {
    if (!parent1 || !parent2 || parent1->num_layers != parent2->num_layers) {
        return NULL;
    }

    NeuralNetwork* child = nn_create(parent1->num_layers, parent1->architecture, parent1->activation_hidden, parent1->activation_output);
    if (!child) return NULL;

    double alpha = (double)rand() / RAND_MAX;

    for (int i = 0; i < parent1->num_layers - 1; i++) {
        // Weights
        for (int r = 0; r < parent1->weights[i]->rows; r++) {
            for (int c = 0; c < parent1->weights[i]->cols; c++) {
                child->weights[i]->data[r][c] = alpha * parent1->weights[i]->data[r][c] + (1 - alpha) * parent2->weights[i]->data[r][c];
            }
        }
        // Biases
        for (int c = 0; c < parent1->biases[i]->cols; c++) {
            child->biases[i]->data[0][c] = alpha * parent1->biases[i]->data[0][c] + (1 - alpha) * parent2->biases[i]->data[0][c];
        }
    }

    return child;
}


NeuralNetwork* crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2, CrossoverType crossover_type) {
    switch (crossover_type) {
        case UNIFORM_CROSSOVER:
            return uniform_crossover(parent1, parent2);
        case SINGLE_POINT_CROSSOVER:
            return single_point_crossover(parent1, parent2);
        case TWO_POINT_CROSSOVER:
            return two_point_crossover(parent1, parent2);
        case ARITHMETIC_CROSSOVER:
            return arithmetic_crossover(parent1, parent2);
        default:
            return uniform_crossover(parent1, parent2);
    }
}
