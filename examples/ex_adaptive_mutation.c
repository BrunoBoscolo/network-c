#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gann.h"

int main() {
    // Seed the random number generator
    srand(time(NULL));

    printf("--- Example: Training with Adaptive Mutation ---\n");

    // --- 1. Load MNIST Data ---
    Dataset* train_dataset = load_mnist_dataset("data/train-images.idx3-ubyte",
                                                "data/train-labels.idx1-ubyte");
    if (!train_dataset) {
        fprintf(stderr, "Failed to load training data.\n");
        return 1;
    }

    // --- 2. Define Training Parameters ---
    const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 64, MNIST_NUM_CLASSES};
    GannTrainParams params = {
        .architecture = ARCHITECTURE,
        .num_layers = sizeof(ARCHITECTURE) / sizeof(int),
        .population_size = 30,
        .num_generations = 50,
        .mutation_rate = 0.5f,
        .mutation_chance = 0.25f,
        .fitness_samples = 500,
        .selection_type = ELITISM_SELECTION,
        .tournament_size = 0,
        .activation_hidden = RELU,
        .crossover_type = UNIFORM_CROSSOVER,
        .mutation_type = ADAPTIVE_MUTATION,
        .mutation_std_dev = 0, // Not used for this mutation type
        .logging = true
    };

    printf("This example demonstrates adaptive mutation.\n");
    printf("Mutation rate adapts based on population diversity (fitness standard deviation).\n\n");

    // --- 3. Run Training ---
    NeuralNetwork* best_net = gann_train(&params, train_dataset, NULL);

    // --- 4. Save the Best Network ---
    if (best_net) {
        printf("--------------------\n");
        if (nn_save(best_net, "ex_adaptive_mutation.dat")) {
            printf("Best network saved to ex_adaptive_mutation.dat\n");
        } else {
            fprintf(stderr, "Failed to save the best network.\n");
        }
        nn_free(best_net);
    } else {
        fprintf(stderr, "Training failed to produce a network.\n");
    }

    // --- 5. Cleanup ---
    free_dataset(train_dataset);

    return 0;
}
