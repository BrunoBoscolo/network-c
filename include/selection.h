#ifndef SELECTION_H
#define SELECTION_H

#include "neural_network.h"
#include "evolution.h" // For NetworkFitness

/**
 * @file selection.h
 * @brief Defines strategies for selecting parent networks in a genetic algorithm.
 * @details Selection is the process of choosing which individuals from the current
 * generation will be parents for the next generation. Fitter individuals have a
 * higher chance of being selected.
 */

/**
 * @brief Enumeration of supported parent selection strategies.
 */
typedef enum {
    ELITISM_SELECTION,        /**< Selects the top N fittest individuals. */
    TOURNAMENT_SELECTION,     /**< Selects winners from random sub-group competitions. */
    ROULETTE_WHEEL_SELECTION, /**< Selects individuals based on a probability proportional to their fitness. */
    RANK_SELECTION            /**< Selects individuals based on their rank, not just raw fitness. */
} SelectionType;

/**
 * @brief Selects a pool of fittest individuals from a population to act as parents.
 * @details This function acts as a dispatcher, calling the appropriate underlying
 * selection method (e.g., tournament, elitism) based on the `selection_type`
 * parameter. It returns a new array containing the selected individuals, which
 * will be used for crossover.
 * @param population_with_fitness An array of `NetworkFitness` structs representing the entire population.
 * @param population_size The total number of individuals in the population.
 * @param num_fittest A pointer to an integer that will be populated with the number of individuals in the returned array.
 * @param selection_type The selection strategy to use (e.g., `TOURNAMENT_SELECTION`).
 * @param tournament_size The number of individuals to compete in a tournament. Only used if `selection_type` is `TOURNAMENT_SELECTION`.
 * @return A new array of `NetworkFitness` structs for the selected parents. The caller is responsible for freeing this array.
 *         The `NeuralNetwork` pointers within the returned structs point to the *original* networks
 *         and should NOT be freed separately, as they are still owned by the main population array.
 * @return `NULL` on failure.
 */
NetworkFitness* select_fittest(NetworkFitness* population_with_fitness, int population_size, int* num_fittest, SelectionType selection_type, int tournament_size);

#endif // SELECTION_H
