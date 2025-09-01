#include "minunit.h"
#include "evolution.h"
#include <math.h>
#include <stdlib.h>

extern const double TEST_EPSILON;

const char* test_crossover() {
    int architecture[] = {2, 2, 1};
    NeuralNetwork* parent1 = create_neural_network(3, architecture);
    NeuralNetwork* parent2 = create_neural_network(3, architecture);

    // Seed rand() for predictable crossover
    srand(42);

    // Manually set weights and biases for parents
    parent1->weights[0]->data[0][0] = 0.1;
    parent2->weights[0]->data[0][0] = 0.3;
    parent1->biases[0]->data[0][0] = 0.5;
    parent2->biases[0]->data[0][0] = 0.7;

    NeuralNetwork* child = crossover(parent1, parent2);
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
