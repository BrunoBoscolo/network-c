#ifndef SELECTION_H
#define SELECTION_H

#include "neural_network.h"
#include "evolution.h" // For NetworkFitness

// Enum for selection strategies
typedef enum {
    ELITISM_SELECTION,
    TOURNAMENT_SELECTION,
    ROULETTE_WHEEL_SELECTION,
    RANK_SELECTION
} SelectionType;

/**
 * @brief Selects the fittest individuals from a population.
 * @param population_with_fitness An array of networks with their fitness scores.
 * @param population_size The size of the population.
 * @param num_fittest A pointer to an integer that will be filled with the number of fittest individuals selected.
 * @param selection_type The selection strategy to use.
 * @param tournament_size The size of the tournament, if using tournament selection.
 * @return An array of the selected fittest networks.
 */
NetworkFitness* select_fittest(NetworkFitness* population_with_fitness, int population_size, int* num_fittest, SelectionType selection_type, int tournament_size);

#endif // SELECTION_H
