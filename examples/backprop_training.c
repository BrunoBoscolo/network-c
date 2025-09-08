#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gann.h"

int main() {
    // Seed the random number generator.
    // Using a fixed seed (e.g., 12345) makes the training process deterministic,
    // which is useful for debugging and for comparing different training runs.
    // To get different results on each run, you could use: gann_seed_rng(time(NULL));
    gann_seed_rng(12345);

    printf("--- Starting MNIST Training with Backpropagation ---\n");

    // --- 1. Load MNIST Data ---
    Dataset* train_dataset = load_mnist_dataset("data/train-images.idx3-ubyte",
                                                "data/train-labels.idx1-ubyte");
    if (!train_dataset) {
        fprintf(stderr, "Error: Failed to load training data. Check file paths and integrity.\n");
        return 1;
    }

    // --- 2. Define Training Parameters ---
    const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};
    GannBackpropParams params = {
        .architecture = ARCHITECTURE,
        .num_layers = sizeof(ARCHITECTURE) / sizeof(int),
        .learning_rate = 0.001, // ADAM works well with smaller learning rates
        .epochs = 5,
        .batch_size = 32,
        .activation_hidden = RELU,
        .activation_output = SIGMOID,
        // --- Optimizer Configuration ---
        .optimizer_type = ADAM, // Choose between SGD, ADAM, RMSPROP
        // ADAM & RMSprop parameters (ignored if optimizer_type is SGD)
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .logging = true // Print epoch progress
    };

    printf("Network architecture: [");
    for (int i = 0; i < params.num_layers; i++)
        printf("%d%s", params.architecture[i], i == params.num_layers - 1 ? "" : ", ");
    printf("]\n");

    // --- 3. Run Training ---
    printf("--------------------\n");
    printf("Starting training...\n");
    NeuralNetwork* net = gann_train_with_backprop(&params, train_dataset);

    // --- 4. Check for errors and Save the Network ---
    if (net) {
        printf("Training complete.\n");
        if (nn_save(net, "trained_network_backprop.dat")) {
            printf("Trained network saved to trained_network_backprop.dat\n");
        } else {
            GannError err = gann_get_last_error();
            fprintf(stderr, "Error: Failed to save the network. Reason: %s\n", gann_error_to_string(err));
        }
        nn_free(net);
    } else {
        GannError err = gann_get_last_error();
        fprintf(stderr, "Error: Training failed. Reason: %s\n", gann_error_to_string(err));
    }

    // --- 5. Cleanup ---
    free_dataset(train_dataset);

    return 0;
}
