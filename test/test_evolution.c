#include "minunit.h"
#include "evolution.h"
#include <math.h>
#include <stdlib.h>

extern const double TEST_EPSILON;

const char* test_crossover() {
    int architecture[] = {2, 2, 1};
    NeuralNetwork* parent1 = create_neural_network(3, architecture, SIGMOID, SIGMOID);
    NeuralNetwork* parent2 = create_neural_network(3, architecture, SIGMOID, SIGMOID);

    // Seed rand() for predictable crossover
    srand(42);

    // Manually set weights and biases for parents
    parent1->weights[0]->data[0][0] = 0.1;
    parent2->weights[0]->data[0][0] = 0.3;
    parent1->biases[0]->data[0][0] = 0.5;
    parent2->biases[0]->data[0][0] = 0.7;

    NeuralNetwork* child = crossover(parent1, parent2, UNIFORM);
    mu_assert("Crossover failed to create a child", child != NULL);

    // Check if the child's weight is from one of the parents
    double child_weight = child->weights[0]->data[0][0];
    int is_from_parent1 = fabs(child_weight - parent1->weights[0]->data[0][0]) < TEST_EPSILON;
    int is_from_parent2 = fabs(child_weight - parent2->weights[0]->data[0][0]) < TEST_EPSILON;
    mu_assert("Child weight is not from either parent", is_from_parent1 || is_from_parent2);

    // Check if the child's bias is from one of the parents
    double child_bias = child->biases[0]->data[0][0];
    is_from_parent1 = fabs(child_bias - parent1->biases[0]->data[0][0]) < TEST_EPSILON;
    is_from_parent2 = fabs(child_bias - parent2->biases[0]->data[0][0]) < TEST_EPSILON;
    mu_assert("Child bias is not from either parent", is_from_parent1 || is_from_parent2);

    free_neural_network(parent1);
    free_neural_network(parent2);
    free_neural_network(child);

    return NULL;
}

const char* test_single_point_crossover() {
    int architecture[] = {2, 2, 1};
    NeuralNetwork* parent1 = create_neural_network(3, architecture, SIGMOID, SIGMOID);
    NeuralNetwork* parent2 = create_neural_network(3, architecture, SIGMOID, SIGMOID);

    // Manually set all weights and biases to be distinct
    for (int i = 0; i < parent1->num_layers - 1; i++) {
        for (int r = 0; r < parent1->weights[i]->rows; r++) {
            for (int c = 0; c < parent1->weights[i]->cols; c++) {
                parent1->weights[i]->data[r][c] = 1.0;
                parent2->weights[i]->data[r][c] = 2.0;
            }
        }
        for (int c = 0; c < parent1->biases[i]->cols; c++) {
            parent1->biases[i]->data[0][c] = 1.0;
            parent2->biases[i]->data[0][c] = 2.0;
        }
    }

    srand(42); // Seed for predictable crossover point
    NeuralNetwork* child = crossover(parent1, parent2, SINGLE_POINT);
    mu_assert("Single-point crossover failed to create a child", child != NULL);

    srand(42);
    int total_weights = 0;
    for (int i = 0; i < parent1->num_layers - 1; i++) {
        total_weights += parent1->weights[i]->rows * parent1->weights[i]->cols;
        total_weights += parent1->biases[i]->cols;
    }

    int crossover_point = rand() % total_weights;
    int current_weight = 0;
    int passed = 1;

    for (int i = 0; i < child->num_layers - 1; i++) {
        for (int r = 0; r < child->weights[i]->rows; r++) {
            for (int c = 0; c < child->weights[i]->cols; c++) {
                double expected = (current_weight < crossover_point) ? 1.0 : 2.0;
                if (fabs(child->weights[i]->data[r][c] - expected) > TEST_EPSILON) {
                    passed = 0;
                }
                current_weight++;
            }
        }
        for (int c = 0; c < child->biases[i]->cols; c++) {
            double expected = (current_weight < crossover_point) ? 1.0 : 2.0;
            if (fabs(child->biases[i]->data[0][c] - expected) > TEST_EPSILON) {
                passed = 0;
            }
            current_weight++;
        }
    }

    mu_assert("Single-point crossover did not work as expected", passed);

    free_neural_network(parent1);
    free_neural_network(parent2);
    free_neural_network(child);

    return NULL;
}

const char* test_two_point_crossover() {
    int architecture[] = {2, 2, 1};
    NeuralNetwork* parent1 = create_neural_network(3, architecture, SIGMOID, SIGMOID);
    NeuralNetwork* parent2 = create_neural_network(3, architecture, SIGMOID, SIGMOID);

    // Manually set all weights and biases to be distinct
    for (int i = 0; i < parent1->num_layers - 1; i++) {
        for (int r = 0; r < parent1->weights[i]->rows; r++) {
            for (int c = 0; c < parent1->weights[i]->cols; c++) {
                parent1->weights[i]->data[r][c] = 1.0;
                parent2->weights[i]->data[r][c] = 2.0;
            }
        }
        for (int c = 0; c < parent1->biases[i]->cols; c++) {
            parent1->biases[i]->data[0][c] = 1.0;
            parent2->biases[i]->data[0][c] = 2.0;
        }
    }

    srand(42); // Seed for predictable crossover points
    NeuralNetwork* child = crossover(parent1, parent2, TWO_POINT);
    mu_assert("Two-point crossover failed to create a child", child != NULL);

    srand(42); // Reset seed to get the same points for checking
    int total_weights = 0;
    for (int i = 0; i < parent1->num_layers - 1; i++) {
        total_weights += parent1->weights[i]->rows * parent1->weights[i]->cols;
        total_weights += parent1->biases[i]->cols;
    }

    int point1 = rand() % total_weights;
    int point2 = rand() % total_weights;
    if (point1 > point2) {
        int temp = point1;
        point1 = point2;
        point2 = temp;
    }

    int current_weight = 0;
    int passed = 1;

    for (int i = 0; i < child->num_layers - 1; i++) {
        for (int r = 0; r < child->weights[i]->rows; r++) {
            for (int c = 0; c < child->weights[i]->cols; c++) {
                double expected = (current_weight >= point1 && current_weight < point2) ? 2.0 : 1.0;
                if (fabs(child->weights[i]->data[r][c] - expected) > TEST_EPSILON) {
                    passed = 0;
                }
                current_weight++;
            }
        }
        for (int c = 0; c < child->biases[i]->cols; c++) {
            double expected = (current_weight >= point1 && current_weight < point2) ? 2.0 : 1.0;
            if (fabs(child->biases[i]->data[0][c] - expected) > TEST_EPSILON) {
                passed = 0;
            }
            current_weight++;
        }
    }

    mu_assert("Two-point crossover did not work as expected", passed);

    free_neural_network(parent1);
    free_neural_network(parent2);
    free_neural_network(child);

    return NULL;
}
