#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gann.h"
#include "utils.h"

int main() {
    gann_seed_rng(12345);

    printf("--- Starting MNIST Training with Backpropagation (Progressive Epochs) ---\n");

    // --- 1. Load MNIST Data ---
    const char* data_prefix = find_data_path_prefix();
    char train_images_path[256];
    char train_labels_path[256];
    char test_images_path[256];
    char test_labels_path[256];

    snprintf(train_images_path, sizeof(train_images_path), "%s%s", data_prefix, "train-images.idx3-ubyte");
    snprintf(train_labels_path, sizeof(train_labels_path), "%s%s", data_prefix, "train-labels.idx1-ubyte");
    snprintf(test_images_path, sizeof(test_images_path), "%s%s", data_prefix, "t10k-images.idx3-ubyte");
    snprintf(test_labels_path, sizeof(test_labels_path), "%s%s", data_prefix, "t10k-labels.idx1-ubyte");

    Dataset* train_dataset = load_mnist_dataset(train_images_path, train_labels_path);
    Dataset* test_dataset = load_mnist_dataset(test_images_path, test_labels_path);
    if (!train_dataset || !test_dataset) {
        fprintf(stderr, "Error: Failed to load MNIST data. Check file paths and integrity.\n");
        return 1;
    }

    // --- 2. Define Training Parameters ---
    const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};
    GannBackpropParams params = {
        .architecture = ARCHITECTURE,
        .num_layers = sizeof(ARCHITECTURE) / sizeof(int),
        .learning_rate = 0.001,
        .batch_size = 32,
        .activation_hidden = RELU,
        .activation_output = SIGMOID,
        .optimizer_type = ADAM,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .logging = false, // Disable logging for cleaner output
        .early_stopping_patience = 10,
        .early_stopping_threshold = 0.001
    };

    // --- 3. Split Data ---
    Dataset* validation_dataset = malloc(sizeof(Dataset));
    Dataset* new_train_dataset = malloc(sizeof(Dataset));
    split_dataset(train_dataset, 10000, new_train_dataset, validation_dataset);
    printf("Training data: %d samples | Validation data: %d samples\n", new_train_dataset->num_items, validation_dataset->num_items);

    // --- 4. Progressive Epoch Training ---
    const int EPOCH_COUNTS[] = {1, 5, 10, 20, 50};
    int num_epoch_counts = sizeof(EPOCH_COUNTS) / sizeof(int);
    FILE* results_file = fopen("backprop_progressive_epochs.dat", "w");
    if (!results_file) {
        fprintf(stderr, "Error: Could not open results file.\n");
        return 1;
    }

    for (int i = 0; i < num_epoch_counts; i++) {
        params.epochs = EPOCH_COUNTS[i];
        printf("\n--- Training with %d epochs ---\n", params.epochs);

        // Reset the seed for reproducibility of each run
        gann_seed_rng(12345);

        NeuralNetwork* net = gann_train_with_backprop(&params, new_train_dataset, validation_dataset);

        if (net) {
            double accuracy = gann_evaluate(net, test_dataset);
            printf("Accuracy after %d epochs: %.4f\n", params.epochs, accuracy);
            fprintf(results_file, "%d %.4f\n", params.epochs, accuracy);
            nn_free(net);
        } else {
            GannError err = gann_get_last_error();
            fprintf(stderr, "Error: Training failed for %d epochs. Reason: %s\n", params.epochs, gann_error_to_string(err));
        }
    }

    // --- 5. Cleanup ---
    fclose(results_file);
    free_dataset(train_dataset);
    free_dataset(test_dataset);
    free(new_train_dataset);
    free(validation_dataset);

    printf("\nProgressive epoch training complete. Results saved to backprop_progressive_epochs.dat\n");

    return 0;
}
