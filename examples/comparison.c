#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gann.h"
#include "utils.h"

// Helper function to print time elapsed
void print_time_elapsed(clock_t start, clock_t end) {
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Training time: %f seconds\n", time_spent);
}

int main() {
    // Seed the random number generator
    srand(time(NULL));

    printf("--- Comparing Genetic Algorithm vs. Backpropagation ---\n\n");

    // --- 1. Load Data ---
    printf("Loading MNIST dataset...\n");
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
        fprintf(stderr, "Failed to load MNIST data.\n");
        return 1;
    }
    printf("Dataset loaded successfully.\n\n");

    // --- 2. Define Common Network Architecture ---
    const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};
    const int NUM_LAYERS = sizeof(ARCHITECTURE) / sizeof(int);

    printf("Network Architecture: [");
    for (int i = 0; i < NUM_LAYERS; i++) printf("%d%s", ARCHITECTURE[i], i == NUM_LAYERS - 1 ? "" : ", ");
    printf("]\n\n");

    // --- 3. Genetic Algorithm Training ---
    printf("--- Training with Genetic Algorithm ---\n");
    GannTrainParams ga_params = {
        .architecture = ARCHITECTURE,
        .num_layers = NUM_LAYERS,
        .population_size = 50,
        .num_generations = 100,
        .mutation_rate = 0.5f,
        .mutation_chance = 0.25f,
        .fitness_samples = 1000,
        .selection_type = TOURNAMENT_SELECTION,
        .tournament_size = 4,
        .activation_hidden = LEAKY_RELU,
        .crossover_type = UNIFORM_CROSSOVER,
        .mutation_type = UNIFORM_MUTATION,
        .logging = false
    };

    ga_params.early_stopping_patience = 0; // Disabled for this comparison

    clock_t ga_start = clock();
    NeuralNetwork* ga_net = gann_train(&ga_params, train_dataset, NULL);
    clock_t ga_end = clock();

    if (ga_net) {
        double ga_accuracy = gann_evaluate(ga_net, test_dataset);
        printf("GA Final Accuracy: %.2f%%\n", ga_accuracy * 100);
        print_time_elapsed(ga_start, ga_end);
        nn_free(ga_net);
    } else {
        fprintf(stderr, "GA Training failed.\n");
    }
    printf("\n");


    // --- 4. Backpropagation Training ---
    printf("--- Training with Backpropagation ---\n");
    GannBackpropParams bp_params = {
        .architecture = ARCHITECTURE,
        .num_layers = NUM_LAYERS,
        .learning_rate = 0.01,
        .epochs = 10,
        .batch_size = 32,
        .activation_hidden = RELU,
        .activation_output = SIGMOID
    };

    bp_params.early_stopping_patience = 0; // Disabled for this comparison
    bp_params.logging = false;

    clock_t bp_start = clock();
    NeuralNetwork* bp_net = gann_train_with_backprop(&bp_params, train_dataset, NULL);
    clock_t bp_end = clock();

    if (bp_net) {
        double bp_accuracy = gann_evaluate(bp_net, test_dataset);
        printf("Backprop Final Accuracy: %.2f%%\n", bp_accuracy * 100);
        print_time_elapsed(bp_start, bp_end);
        nn_free(bp_net);
    } else {
        fprintf(stderr, "Backprop Training failed.\n");
    }
    printf("\n");


    // --- 5. Cleanup ---
    free_dataset(train_dataset);
    free_dataset(test_dataset);

    return 0;
}
