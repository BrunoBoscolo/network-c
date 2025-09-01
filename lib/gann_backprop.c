#include "gann.h"
#include <stdio.h>
#include <stdlib.h>

NeuralNetwork* gann_train_with_backprop(const GannBackpropParams* params, const Dataset* train_dataset) {
    printf("--- Starting Backpropagation Training ---\n");

    // 1. Create the Neural Network
    NeuralNetwork* net = create_neural_network(
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
    initialize_network(net);

    // 3. Start the training process
    printf("Training with parameters:\n");
    printf("  Learning Rate: %f\n", params->learning_rate);
    printf("  Epochs: %d\n", params->epochs);
    printf("  Batch Size: %d\n", params->batch_size);

    backpropagate(
        net,
        train_dataset,
        params->learning_rate,
        params->epochs,
        params->batch_size
    );

    printf("--- Backpropagation Training Finished ---\n");

    // 4. Return the trained network
    return net;
}
