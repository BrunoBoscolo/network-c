#include "minunit.h"
#include "../include/evolution.h"
#include <math.h>

extern const double TEST_EPSILON;

const char* test_crossover() {
    int architecture[] = {2, 2, 1};
    NeuralNetwork* parent1 = create_neural_network(3, architecture);
    NeuralNetwork* parent2 = create_neural_network(3, architecture);

    // Manually set weights and biases for parents
    parent1->weights[0]->data[0][0] = 0.1;
    parent2->weights[0]->data[0][0] = 0.3;
    parent1->biases[0]->data[0][0] = 0.5;
    parent2->biases[0]->data[0][0] = 0.7;

    NeuralNetwork* child = crossover(parent1, parent2);
    mu_assert("Crossover failed to create a child", child != NULL);

    // Check that child's weight is the average of parents'
    double expected_weight = (0.1 + 0.3) / 2.0;
    double actual_weight = child->weights[0]->data[0][0];
    mu_assert("Crossover weight calculation is incorrect", fabs(actual_weight - expected_weight) < TEST_EPSILON);

    // Check that child's bias is the average of parents'
    double expected_bias = (0.5 + 0.7) / 2.0;
    double actual_bias = child->biases[0]->data[0][0];
    mu_assert("Crossover bias calculation is incorrect", fabs(actual_bias - expected_bias) < TEST_EPSILON);

    free_neural_network(parent1);
    free_neural_network(parent2);
    free_neural_network(child);

    return NULL;
}
