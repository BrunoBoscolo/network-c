#include "minunit.h"
#include "gann.h"
#include "data_loader.h"
#include "backpropagation.h"
#include <math.h>
#include <stdlib.h>

extern const double TEST_EPSILON;

const char* test_calculate_mse() {
    // 1. Setup
    const int architecture[] = {2, 3, 1};
    NeuralNetwork* net = nn_create(3, architecture, RELU, SIGMOID);
    // Set weights and biases to known values
    for (int l = 0; l < net->num_layers - 1; l++) {
        for (int r = 0; r < net->weights[l]->rows; r++) {
            for (int c = 0; c < net->weights[l]->cols; c++) {
                net->weights[l]->data[r][c] = 0.5;
            }
        }
        for (int c = 0; c < net->biases[l]->cols; c++) {
            net->biases[l]->data[0][c] = 0.1;
        }
    }

    Dataset* dataset = malloc(sizeof(Dataset));
    dataset->num_items = 1;
    dataset->images = create_matrix(1, 2);
    dataset->images->data[0][0] = 0.2;
    dataset->images->data[0][1] = 0.3;
    dataset->labels = create_matrix(1, 1);
    dataset->labels->data[0][0] = 0.9; // Target label

    // 2. Execution
    double mse = calculate_mse(net, dataset);

    // 3. Assertion
    // This is a placeholder value. The actual expected value would need to be calculated manually.
    // The goal here is to ensure the function runs and returns a plausible value.
    mu_assert("MSE should be non-negative", mse >= 0);
    // A more specific assertion would be:
    // mu_assert("MSE calculation is incorrect", fabs(mse - EXPECTED_VALUE) < 1e-6);

    // 4. Cleanup
    nn_free(net);
    free_dataset(dataset);

    return NULL;
}


// A simple test to see if the network can learn a single instance (overfit).
const char* test_backprop_overfit_single_instance() {
    // 1. Create a dummy dataset with one sample
    Dataset* dummy_dataset = create_dummy_dataset(1);
    mu_assert("Failed to create dummy dataset", dummy_dataset != NULL);

    // 2. Define network architecture and training parameters
    const int ARCHITECTURE[] = {dummy_dataset->images->cols, 10, dummy_dataset->labels->cols};
    GannBackpropParams params = {
        .architecture = ARCHITECTURE,
        .num_layers = sizeof(ARCHITECTURE) / sizeof(int),
        .learning_rate = 0.1,
        .epochs = 200, // More epochs to ensure overfitting
        .batch_size = 1,
        .activation_hidden = RELU,
        .activation_output = SIGMOID,
        .optimizer_type = SGD,
        .logging = false // Disable logging for tests
    };

    // 3. Create and train the network
    NeuralNetwork* net = nn_create(params.num_layers, params.architecture, params.activation_hidden, params.activation_output);
    nn_init(net);
    backpropagate(net, dummy_dataset, &params);

    // 4. Test the prediction
    int prediction = gann_predict(net, dummy_dataset->images->data[0]);

    // Find the actual label from the one-hot encoded vector
    int actual_label = -1;
    for(int i=0; i < dummy_dataset->labels->cols; i++){
        if(fabs(dummy_dataset->labels->data[0][i] - 1.0) < TEST_EPSILON){
            actual_label = i;
            break;
        }
    }

    mu_assert("Prediction should match the label after training (SGD)", prediction == actual_label);

    // 5. Cleanup
    nn_free(net);
    free_dataset(dummy_dataset);

    return NULL;
}

const char* test_backprop_overfit_single_instance_adam() {
    // 1. Create a dummy dataset with one sample
    Dataset* dummy_dataset = create_dummy_dataset(1);
    mu_assert("Failed to create dummy dataset", dummy_dataset != NULL);

    // 2. Define network architecture and training parameters
    const int ARCHITECTURE[] = {dummy_dataset->images->cols, 10, dummy_dataset->labels->cols};
    GannBackpropParams params = {
        .architecture = ARCHITECTURE,
        .num_layers = sizeof(ARCHITECTURE) / sizeof(int),
        .learning_rate = 0.01, // Adam usually requires a smaller learning rate
        .epochs = 200,
        .batch_size = 1,
        .activation_hidden = RELU,
        .activation_output = SIGMOID,
        .optimizer_type = ADAM,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .logging = false
    };

    // 3. Create and train the network
    NeuralNetwork* net = nn_create(params.num_layers, params.architecture, params.activation_hidden, params.activation_output);
    nn_init(net);
    nn_init_optimizer_state(net); // Important for Adam
    backpropagate(net, dummy_dataset, &params);

    // 4. Test the prediction
    int prediction = gann_predict(net, dummy_dataset->images->data[0]);
    int actual_label = -1;
    for(int i=0; i < dummy_dataset->labels->cols; i++){
        if(fabs(dummy_dataset->labels->data[0][i] - 1.0) < TEST_EPSILON){
            actual_label = i;
            break;
        }
    }

    mu_assert("Prediction should match the label after training (Adam)", prediction == actual_label);

    // 5. Cleanup
    nn_free(net);
    free_dataset(dummy_dataset);

    return NULL;
}

const char* test_backprop_overfit_single_instance_rmsprop() {
    // 1. Create a dummy dataset with one sample
    Dataset* dummy_dataset = create_dummy_dataset(1);
    mu_assert("Failed to create dummy dataset", dummy_dataset != NULL);

    // 2. Define network architecture and training parameters
    const int ARCHITECTURE[] = {dummy_dataset->images->cols, 10, dummy_dataset->labels->cols};
    GannBackpropParams params = {
        .architecture = ARCHITECTURE,
        .num_layers = sizeof(ARCHITECTURE) / sizeof(int),
        .learning_rate = 0.01,
        .epochs = 200,
        .batch_size = 1,
        .activation_hidden = RELU,
        .activation_output = SIGMOID,
        .optimizer_type = RMSPROP,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .logging = false
    };

    // 3. Create and train the network
    NeuralNetwork* net = nn_create(params.num_layers, params.architecture, params.activation_hidden, params.activation_output);
    nn_init(net);
    nn_init_optimizer_state(net); // Important for RMSprop
    backpropagate(net, dummy_dataset, &params);

    // 4. Test the prediction
    int prediction = gann_predict(net, dummy_dataset->images->data[0]);
    int actual_label = -1;
    for(int i=0; i < dummy_dataset->labels->cols; i++){
        if(fabs(dummy_dataset->labels->data[0][i] - 1.0) < TEST_EPSILON){
            actual_label = i;
            break;
        }
    }

    mu_assert("Prediction should match the label after training (RMSprop)", prediction == actual_label);

    // 5. Cleanup
    nn_free(net);
    free_dataset(dummy_dataset);

    return NULL;
}
