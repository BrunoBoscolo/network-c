#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gann.h"

// Helper function to convert activation type enum to string for printing
const char* activation_to_string(ActivationType type) {
    switch (type) {
        case SIGMOID: return "Sigmoid";
        case RELU: return "ReLU";
        case LEAKY_RELU: return "Leaky ReLU";
        default: return "Unknown";
    }
}

int main() {
    // Seed the random number generator
    srand(time(NULL));

    printf("--- Comparing Activation Functions ---\n");

    // --- 1. Load MNIST Data ---
    Dataset* train_dataset = load_mnist_dataset("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
    if (!train_dataset) {
        fprintf(stderr, "Failed to load training data.\n");
        return 1;
    }
    Dataset* test_dataset = load_mnist_dataset("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
    if (!test_dataset) {
        fprintf(stderr, "Failed to load test data.\n");
        free_dataset(train_dataset);
        return 1;
    }

    // --- 2. Define Network and Training Parameters ---
    const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};
    ActivationType activations[] = {SIGMOID, RELU, LEAKY_RELU};
    int num_activations = sizeof(activations) / sizeof(ActivationType);

    // --- 3. Loop through each activation function ---
    for (int i = 0; i < num_activations; i++) {
        ActivationType current_activation = activations[i];
        printf("\n--- Training with %s activation ---\n", activation_to_string(current_activation));

        GannTrainParams params = {
            .architecture = ARCHITECTURE,
            .num_layers = sizeof(ARCHITECTURE) / sizeof(int),
            .population_size = 20, // Smaller population for quicker comparison
            .num_generations = 30, // Fewer generations for quicker comparison
            .mutation_rate = 0.5f,
            .mutation_chance = 0.25f,
            .fitness_samples = 1000,
            .selection_type = TOURNAMENT_SELECTION,
            .tournament_size = 4,
            .activation_hidden = current_activation,
            .activation_output = SIGMOID, // Use sigmoid for output layer
            .crossover_type = UNIFORM_CROSSOVER,
            .mutation_type = UNIFORM_MUTATION,
            .logging = false
        };

        // Train the network
        NeuralNetwork* best_net = gann_train(&params, train_dataset, NULL);
        if (!best_net) {
            fprintf(stderr, "Training failed for %s activation.\n", activation_to_string(current_activation));
            continue; // Skip to the next activation function
        }

        // Evaluate the network
        double accuracy = gann_evaluate(best_net, test_dataset);
        printf("--------------------\n");
        printf("Final accuracy with %s: %.2f%%\n", activation_to_string(current_activation), accuracy * 100.0);
        printf("--------------------\n");

        // Cleanup
        nn_free(best_net);
    }

    // --- 4. Final Cleanup ---
    free_dataset(train_dataset);
    free_dataset(test_dataset);

    return 0;
}
