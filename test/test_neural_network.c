#include "minunit.h"
#include "neural_network.h"
#include "mutation.h"
#include "gann_errors.h"
#include <math.h>

extern const double TEST_EPSILON;

// Test for neural network creation
const char* test_nn_creation() {
    int architecture[] = {2, 2, 1};
    NeuralNetwork* net = nn_create(3, architecture, SIGMOID, SIGMOID);
    mu_assert("nn_create should not return NULL on success", net != NULL);
    mu_assert("nn_create should set GANN_SUCCESS on success", gann_get_last_error() == GANN_SUCCESS);

    nn_init(net);
    mu_assert("nn_init should set GANN_SUCCESS on success", gann_get_last_error() == GANN_SUCCESS);

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

    nn_free(net);
    return NULL;
}

// Test for the forward pass
const char* test_nn_forward_pass() {
    int architecture[] = {2, 2, 1};
    NeuralNetwork* net = nn_create(3, architecture, SIGMOID, SIGMOID);

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

    Matrix* output = nn_forward_pass(net, input);
    mu_assert("Forward pass returned NULL", output != NULL);
    mu_assert("nn_forward_pass should set GANN_SUCCESS on success", gann_get_last_error() == GANN_SUCCESS);

    double actual_output = output->data[0][0];
    mu_assert("Forward pass calculation is incorrect", fabs(actual_output - expected_output) < 1e-4);

    nn_free(net);
    free_matrix(input);
    free_matrix(output);

    return NULL;
}

// Test for neural network error handling
const char* test_nn_errors() {
    // Test nn_create with invalid architecture
    NeuralNetwork* net = nn_create(1, (int[]){2}, SIGMOID, SIGMOID);
    mu_assert("nn_create should fail for < 2 layers", net == NULL);
    mu_assert("nn_create should set GANN_ERROR_INVALID_ARCHITECTURE", gann_get_last_error() == GANN_ERROR_INVALID_ARCHITECTURE);

    // Test nn_create with NULL architecture
    net = nn_create(3, NULL, SIGMOID, SIGMOID);
    mu_assert("nn_create should fail for NULL architecture", net == NULL);
    mu_assert("nn_create should set GANN_ERROR_NULL_ARGUMENT", gann_get_last_error() == GANN_ERROR_NULL_ARGUMENT);

    // Test nn_forward_pass with NULL net
    Matrix* input = create_matrix(1, 2);
    Matrix* output = nn_forward_pass(NULL, input);
    mu_assert("nn_forward_pass should fail for NULL net", output == NULL);
    mu_assert("nn_forward_pass should set GANN_ERROR_NULL_ARGUMENT for NULL net", gann_get_last_error() == GANN_ERROR_NULL_ARGUMENT);

    // Test nn_forward_pass with mismatched dimensions
    int arch[] = {2, 1};
    net = nn_create(2, arch, SIGMOID, SIGMOID);
    Matrix* wrong_input = create_matrix(1, 3);
    output = nn_forward_pass(net, wrong_input);
    mu_assert("nn_forward_pass should fail for mismatched input dimensions", output == NULL);
    mu_assert("nn_forward_pass should set GANN_ERROR_INVALID_DIMENSIONS", gann_get_last_error() == GANN_ERROR_INVALID_DIMENSIONS);

    free_matrix(input);
    free_matrix(wrong_input);
    nn_free(net);

    // Test nn_clone with NULL net
    NeuralNetwork* clone = nn_clone(NULL);
    mu_assert("nn_clone should fail for NULL net", clone == NULL);
    mu_assert("nn_clone should set GANN_ERROR_NULL_ARGUMENT for NULL net", gann_get_last_error() == GANN_ERROR_NULL_ARGUMENT);

    return NULL;
}

const char* test_gaussian_mutation() {
    int architecture[] = {2, 2, 1};
    NeuralNetwork* net = nn_create(3, architecture, SIGMOID, SIGMOID);
    nn_init(net);

    NeuralNetwork* net_clone = nn_clone(net);

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

    nn_free(net);
    nn_free(net_clone);

    return NULL;
}
