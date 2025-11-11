#include "evolution.h"
#include "crossover.h"
#include "gann_errors.h"
#include <stdlib.h>
#include <stdio.h>

// --- Evolution Functions Implementation ---

// Creates an initial population of neural networks
NeuralNetwork** evo_create_initial_population(int population_size, int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output) {
    if (architecture == NULL) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    NeuralNetwork** population = (NeuralNetwork**)malloc(population_size * sizeof(NeuralNetwork*));
    if (!population) {
        gann_set_error(GANN_ERROR_ALLOC_FAILED);
        return NULL;
    }

    for (int i = 0; i < population_size; i++) {
        population[i] = nn_create(num_layers, architecture, activation_hidden, activation_output);
        if (population[i] == NULL) {
            // nn_create sets the error, but we need to clean up
            for (int j = 0; j < i; j++) {
                nn_free(population[j]);
            }
            free(population);
            return NULL;
        }
        nn_init(population[i]);
    }
    return population;
}



// Helper function to select a parent using tournament selection
static int select_parent_tournament(const NetworkFitness* candidates, int num_candidates, int tournament_size) {
    int best_index = -1;
    double best_fitness = -1.0;

    for (int i = 0; i < tournament_size; i++) {
        int competitor_index = rand() % num_candidates;
        if (candidates[competitor_index].fitness > best_fitness) {
            best_fitness = candidates[competitor_index].fitness;
            best_index = competitor_index;
        }
    }
    return best_index;
}


// Creates a new generation using crossover
NeuralNetwork** evo_reproduce(const NetworkFitness* fittest_networks, int num_fittest, int new_population_size, CrossoverType crossover_type, int tournament_size) {
    if (fittest_networks == NULL) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    if (num_fittest == 0 || tournament_size <= 0) {
        gann_set_error(GANN_ERROR_INVALID_PARAM);
        return NULL;
    }

    NeuralNetwork** new_population = (NeuralNetwork**)malloc(new_population_size * sizeof(NeuralNetwork*));
    if (!new_population) {
        gann_set_error(GANN_ERROR_ALLOC_FAILED);
        return NULL;
    }

    for (int i = 0; i < new_population_size; i++) {
        // Choose two parents using tournament selection
        int parent1_index = select_parent_tournament(fittest_networks, num_fittest, tournament_size);
        int parent2_index = select_parent_tournament(fittest_networks, num_fittest, tournament_size);

        const NeuralNetwork* parent1 = fittest_networks[parent1_index].network;
        const NeuralNetwork* parent2 = fittest_networks[parent2_index].network;

        // Create a child using crossover
        NeuralNetwork* child = crossover(parent1, parent2, crossover_type);
        if (!child) {
            for (int j = 0; j < i; j++) {
                nn_free(new_population[j]);
            }
            free(new_population);
            return NULL;
        }

        new_population[i] = child;
    }

    return new_population;
}
