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
    // For this example, we will use the convenient `gann_create_default_params`
    // function to get a struct with sensible default values.
    GannTrainParams params = gann_create_default_params();

    // The two most important parameters that MUST be set by the user are the
    // network architecture and the number of layers.
    const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};
    params.architecture = ARCHITECTURE;
    params.num_layers = sizeof(ARCHITECTURE) / sizeof(int);

    // We can also override any of the default parameters if we want to experiment.
    // For example, let's use a different activation function for the hidden layers
    // and run for fewer generations for a quicker example.
    params.activation_hidden = LEAKY_RELU;
    params.num_generations = 50; // Default is 100

    // Print the final parameters to the console.
    printf("Network architecture: [");
    for (int i = 0; i < params.num_layers; i++)
        printf("%d%s", params.architecture[i], i == params.num_layers - 1 ? "" : ", ");
    printf("]\n");
    printf("Generations: %d | Population: %d | Mutation Chance: %.2f%%\n",
           params.num_generations, params.population_size, params.mutation_chance * 100);


    // --- 3. Run Training ---
    // This single function call encapsulates the entire genetic algorithm process:
    // - Creates an initial population of random neural networks.
    // - For each generation:
    //   - Evaluates the fitness of each network.
    //   - Selects the best networks to be parents.
    //   - Creates a new generation through crossover and mutation.
    // - Returns the best network found after all generations are complete.
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
