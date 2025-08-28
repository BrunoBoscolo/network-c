#include "minunit.h"
#include "../include/neural_network.h"
#include <stdio.h>
#include <math.h>

extern const double TEST_EPSILON;

const char* test_save_and_load_network() {
    int architecture[] = {2, 3, 1};
    NeuralNetwork* original_net = create_neural_network(3, architecture);

    // Set some specific values to test
    original_net->weights[0]->data[0][0] = 0.123;
    original_net->biases[0]->data[0][0] = 0.456;

    const char* filepath = "test_network.dat";
    int result = save_network(original_net, filepath);
    mu_assert("Failed to save network", result == 1);

    NeuralNetwork* loaded_net = load_network(filepath);
    mu_assert("Failed to load network", loaded_net != NULL);

    // Compare architecture
    mu_assert("Loaded network has wrong number of layers", original_net->num_layers == loaded_net->num_layers);
    for (int i = 0; i < original_net->num_layers; i++) {
        mu_assert("Loaded network has wrong architecture", original_net->architecture[i] == loaded_net->architecture[i]);
    }

    // Compare weights and biases
    for (int i = 0; i < original_net->num_layers - 1; i++) {
        for (int r = 0; r < original_net->weights[i]->rows; r++) {
            for (int c = 0; c < original_net->weights[i]->cols; c++) {
                double diff = fabs(original_net->weights[i]->data[r][c] - loaded_net->weights[i]->data[r][c]);
                mu_assert("Loaded network has wrong weights", diff < TEST_EPSILON);
            }
        }
        for (int c = 0; c < original_net->biases[i]->cols; c++) {
            double diff = fabs(original_net->biases[i]->data[0][c] - loaded_net->biases[i]->data[0][c]);
            mu_assert("Loaded network has wrong biases", diff < TEST_EPSILON);
        }
    }

    free_neural_network(original_net);
    free_neural_network(loaded_net);
    remove(filepath);

    return NULL;
}
