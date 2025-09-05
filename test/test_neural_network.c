#include "minunit.h"
#include "neural_network.h"
#include "mutation.h"
#include <math.h>

extern const double TEST_EPSILON;

// Test for neural network creation
const char* test_nn_creation() {
    int architecture[] = {2, 2, 1};
    NeuralNetwork* net = create_neural_network(3, architecture, SIGMOID, SIGMOID);
    initialize_network(net);

    mu_assert("NN creation failed", net != NULL);
    mu_assert("NN num_layers is wrong", net->num_layers == 3);
    mu_assert("NN architecture[0] is wrong", net->architecture[0] == 2);
    mu_assert("NN architecture[1] is wrong", net->architecture[1] == 2);
    mu_assert("NN architecture[2] is wrong", net->architecture[2] == 1);
    mu_assert("NN weights array is null", net->weights != NULL);
    mu_assert("NN biases array is null", net->biases != NULL);

    // Check layer dimensions
    mu_assert("Weights[0] rows incorrect", net->weights[0]->rows == 2);
    mu_assert("Weights[0] cols incorrect", net->weights[0]->cols == 2);
    mu_assert("Weights[1] rows incorrect", net->weights[1]->rows == 2);
    mu_assert("Weights[1] cols incorrect", net->weights[1]->cols == 1);

    mu_assert("Biases[0] rows incorrect", net->biases[0]->rows == 1);
    mu_assert("Biases[0] cols incorrect", net->biases[0]->cols == 2);
    mu_assert("Biases[1] rows incorrect", net->biases[1]->rows == 1);
    mu_assert("Biases[1] cols incorrect", net->biases[1]->cols == 1);

    free_neural_network(net);
    return NULL;
}

// Test for the forward pass
const char* test_nn_forward_pass() {
    int architecture[] = {2, 2, 1};
    NeuralNetwork* net = create_neural_network(3, architecture, SIGMOID, SIGMOID);

    // Manually set weights and biases for a predictable outcome
    // Layer 1 weights: [[0.1, 0.2], [0.3, 0.4]]
    net->weights[0]->data[0][0] = 0.1; net->weights[0]->data[0][1] = 0.2;
    net->weights[0]->data[1][0] = 0.3; net->weights[0]->data[1][1] = 0.4;
    // Layer 1 biases: [[0.5, 0.5]]
    net->biases[0]->data[0][0] = 0.5; net->biases[0]->data[0][1] = 0.5;

    // Layer 2 weights: [[0.5], [0.6]]
    net->weights[1]->data[0][0] = 0.5;
    net->weights[1]->data[1][0] = 0.6;
    // Layer 2 biases: [[-0.5]]
    net->biases[1]->data[0][0] = -0.5;

    // Input: [1, 1]
    Matrix* input = create_matrix(1, 2);
    input->data[0][0] = 1;
    input->data[0][1] = 1;

    // --- Expected output calculation ---
    // Hidden layer input:
    //   h_in1 = (1*0.1 + 1*0.3) + 0.5 = 0.9
    //   h_in2 = (1*0.2 + 1*0.4) + 0.5 = 1.1
    // Hidden layer output (after sigmoid):
    //   h_out1 = sigmoid(0.9) = 0.7109495
    //   h_out2 = sigmoid(1.1) = 0.7502601
    // Output layer input:
    //   o_in = (h_out1*0.5 + h_out2*0.6) - 0.5
    //        = (0.7109*0.5 + 0.7502*0.6) - 0.5 = 0.30559
    // Output after sigmoid:
    //   o_out = sigmoid(0.30559) = 0.5758
    double expected_output = 0.5758;

    Matrix* output = forward_pass(net, input);
    mu_assert("Forward pass returned NULL", output != NULL);

    double actual_output = output->data[0][0];
    mu_assert("Forward pass calculation is incorrect", fabs(actual_output - expected_output) < 1e-4);

    free_neural_network(net);
    free_matrix(input);
    free_matrix(output);

    return NULL;
}

const char* test_gaussian_mutation() {
    int architecture[] = {2, 2, 1};
    NeuralNetwork* net = create_neural_network(3, architecture, SIGMOID, SIGMOID);
    initialize_network(net);

    NeuralNetwork* net_clone = clone_network(net);

    srand(42);
    mutate_network(net, 0.1f, 1.0f, GAUSSIAN_MUTATION, 0.2, 0, 0, 0); // 100% chance of mutation

    int changed = 0;
    for (int i = 0; i < net->num_layers - 1; i++) {
        for (int r = 0; r < net->weights[i]->rows; r++) {
            for (int c = 0; c < net->weights[i]->cols; c++) {
                if (fabs(net->weights[i]->data[r][c] - net_clone->weights[i]->data[r][c]) > TEST_EPSILON) {
                    changed = 1;
                    break;
                }
            }
            if(changed) break;
        }
        if(changed) break;
    }

    mu_assert("Gaussian mutation did not change network weights", changed);

    free_neural_network(net);
    free_neural_network(net_clone);

    return NULL;
}
