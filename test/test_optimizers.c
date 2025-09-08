#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "minunit.h"
#include "test_suites.h"
#include "../include/gann.h"
#include "../include/neural_network.h"
#include "../include/backpropagation.h"
#include "../include/matrix.h"

// --- Helper Functions ---

// A mock backpropagate function to call the internal update functions
void backpropagate(NeuralNetwork* net, const Dataset* train_dataset, const GannBackpropParams* params);


// --- Test Cases ---

const char* test_rmsprop_update() {
    // 1. Setup
    const int architecture[] = {2, 2};
    NeuralNetwork* net = nn_create(2, architecture, RELU, SIGMOID);
    nn_init(net);
    nn_init_optimizer_state(net);

    GannBackpropParams params = {
        .learning_rate = 0.01,
        .beta2 = 0.9,
        .epsilon = 1e-8
    };

    Matrix* weight_gradient = create_matrix(2, 2);
    weight_gradient->data[0][0] = 0.1;
    weight_gradient->data[0][1] = -0.2;
    weight_gradient->data[1][0] = 0.3;
    weight_gradient->data[1][1] = -0.4;

    Matrix* bias_gradient = create_matrix(1, 2);
    bias_gradient->data[0][0] = 0.05;
    bias_gradient->data[0][1] = -0.15;

    Matrix** weight_gradients = &weight_gradient;
    Matrix** bias_gradients = &bias_gradient;

    double initial_weight = net->weights[0]->data[0][0];

    // 2. Execution
    update_weights_rmsprop(net, weight_gradients, bias_gradients, &params, 1);

    // 3. Assertion
    double grad_w = 0.1;
    double v_w = (1 - params.beta2) * (grad_w * grad_w);
    double expected_weight = initial_weight - (params.learning_rate / (sqrt(v_w) + params.epsilon)) * grad_w;

    mu_assert("RMSprop weight update is incorrect", fabs(net->weights[0]->data[0][0] - expected_weight) < 1e-6);

    // 4. Cleanup
    nn_free(net);
    free_matrix(weight_gradient);
    free_matrix(bias_gradient);

    return 0;
}


const char* test_adam_update() {
    // 1. Setup
    const int architecture[] = {1, 1};
    NeuralNetwork* net = nn_create(2, architecture, RELU, SIGMOID);
    nn_init(net);
    nn_init_optimizer_state(net);
    double initial_weight = net->weights[0]->data[0][0];

    GannBackpropParams params = {
        .learning_rate = 0.001,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8
    };

    Matrix* weight_gradient = create_matrix(1, 1);
    weight_gradient->data[0][0] = 0.5;
    Matrix* bias_gradient = create_matrix(1, 1);
    bias_gradient->data[0][0] = -0.2;
    Matrix** weight_gradients = &weight_gradient;
    Matrix** bias_gradients = &bias_gradient;
    int t = 1;

    // 2. Execution
    update_weights_adam(net, weight_gradients, bias_gradients, &params, 1, t);

    // 3. Assertion
    double grad_w = 0.5;
    double m_w = (1 - params.beta1) * grad_w;
    double v_w = (1 - params.beta2) * (grad_w * grad_w);
    double m_hat = m_w / (1 - pow(params.beta1, t));
    double v_hat = v_w / (1 - pow(params.beta2, t));
    double expected_weight = initial_weight - (params.learning_rate * m_hat) / (sqrt(v_hat) + params.epsilon);

    mu_assert("Adam weight update is incorrect", fabs(net->weights[0]->data[0][0] - expected_weight) < 1e-6);

    // 4. Cleanup
    nn_free(net);
    free_matrix(weight_gradient);
    free_matrix(bias_gradient);

    return 0;
}


// --- Test Suite ---

const char* optimizers_test_suite() {
    mu_run_test(test_rmsprop_update);
    mu_run_test(test_adam_update);
    return 0;
}
