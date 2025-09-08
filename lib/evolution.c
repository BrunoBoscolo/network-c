#include "evolution.h"
#include "crossover.h"
#include <stdlib.h>
#include <stdio.h>

// --- Evolution Functions Implementation ---

// Creates an initial population of neural networks
NeuralNetwork** evo_create_initial_population(int population_size, int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output) {
    if (architecture == NULL) {
        fprintf(stderr, "Error: Cannot create population. Provided architecture is NULL.\n");
        return NULL;
    }
    NeuralNetwork** population = (NeuralNetwork**)malloc(population_size * sizeof(NeuralNetwork*));
    if (!population) return NULL;

    for (int i = 0; i < population_size; i++) {
        population[i] = nn_create(num_layers, architecture, activation_hidden, activation_output);
        if (population[i]) {
            nn_init(population[i]);
        }
        // TODO: Handle allocation failure
    }
    return population;
}



// Creates a new generation using crossover
NeuralNetwork** evo_reproduce(const NetworkFitness* fittest_networks, int num_fittest, int new_population_size, CrossoverType crossover_type) {
    if (fittest_networks == NULL) {
        fprintf(stderr, "Error: Cannot reproduce. Provided fittest_networks is NULL.\n");
        return NULL;
    }
    if (num_fittest == 0) return NULL;

    NeuralNetwork** new_population = (NeuralNetwork**)malloc(new_population_size * sizeof(NeuralNetwork*));
    if (!new_population) return NULL;

    for (int i = 0; i < new_population_size; i++) {
        // Choose two random parents from the fittest networks
        int parent1_index = rand() % num_fittest;
        int parent2_index = rand() % num_fittest;
        const NeuralNetwork* parent1 = fittest_networks[parent1_index].network;
        const NeuralNetwork* parent2 = fittest_networks[parent2_index].network;

        // Create a child using crossover
        NeuralNetwork* child = crossover(parent1, parent2, crossover_type);
        if (!child) {
            // Handle crossover failure, e.g., by cloning one parent
            child = nn_clone(parent1);
        }

        new_population[i] = child;
    }

    return new_population;
}
