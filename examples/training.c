#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gann.h"

int main() {
    // Seed the random number generator
    srand(time(NULL));

    printf("--- Starting MNIST Training with the GANN Simple API ---\n");

    // --- 1. Load MNIST Data ---
    Dataset* train_dataset = load_mnist_dataset("data/train-images.idx3-ubyte",
                                                "data/train-labels.idx1-ubyte");
    if (!train_dataset) {
        fprintf(stderr, "Failed to load training data.\n");
        return 1;
    }

    // --- 2. Define Training Parameters ---
    const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 128, MNIST_NUM_CLASSES};
    GannTrainParams params = {
        .architecture = ARCHITECTURE,
        .num_layers = sizeof(ARCHITECTURE) / sizeof(int),
        .population_size = 50,
        .num_generations = 100, // Reduced for a quicker example run
        .mutation_rate = 0.5f,
        .mutation_chance = 0.25f,
        .fitness_samples = 1000,
        .selection_type = TOURNAMENT_SELECTION,
        .tournament_size = 4,
        .activation_hidden = LEAKY_RELU,
        .activation_output = SIGMOID,
        .crossover_type = TWO_POINT_CROSSOVER,
        .mutation_type = GAUSSIAN_MUTATION,
        .mutation_std_dev = 0.2
    };

    printf("Network architecture: [");
    for (int i = 0; i < params.num_layers; i++)
        printf("%d%s", params.architecture[i], i == params.num_layers - 1 ? "" : ", ");
    printf("]\n");

    // --- 3. Run Training ---
    NeuralNetwork* best_net = gann_train(&params, train_dataset);

    // --- 4. Save the Best Network ---
    if (best_net) {
        printf("--------------------\n");
        if (nn_save(best_net, "trained_network.dat")) {
            printf("Best network saved to trained_network.dat\n");
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
