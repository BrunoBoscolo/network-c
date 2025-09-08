#include "gann.h"
#include <stdio.h>
#include <stdlib.h>

NeuralNetwork* gann_train_with_backprop(const GannBackpropParams* params, const Dataset* train_dataset) {
    if (params == NULL || train_dataset == NULL || params->architecture == NULL) {
        fprintf(stderr, "Error: Cannot train with backprop. Provided params, dataset or architecture is NULL.\n");
        return NULL;
    }
    printf("--- Starting Backpropagation Training ---\n");

    // 1. Create the Neural Network
    NeuralNetwork* net = nn_create(
        params->num_layers,
        params->architecture,
        params->activation_hidden,
        params->activation_output
    );
    if (!net) {
        fprintf(stderr, "Failed to create neural network.\n");
        return NULL;
    }

    // 2. Initialize weights and biases
    nn_init(net);

    // 3. Initialize optimizer state
    if (!nn_init_optimizer_state(net)) {
        fprintf(stderr, "Failed to initialize optimizer state.\n");
        nn_free(net);
        return NULL;
    }

    // 4. Start the training process
    printf("Training with parameters:\n");
    printf("  Learning Rate: %f\n", params->learning_rate);
    printf("  Epochs: %d\n", params->epochs);
    printf("  Batch Size: %d\n", params->batch_size);

    backpropagate(net, train_dataset, params);

    printf("--- Backpropagation Training Finished ---\n");

    // 4. Return the trained network
    return net;
}
