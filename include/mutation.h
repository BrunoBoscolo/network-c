#ifndef MUTATION_H
#define MUTATION_H

#include "neural_network.h"

/**
 * @file mutation.h
 * @brief Defines strategies for mutating neural networks.
 * @details Mutation is a genetic operator used to maintain genetic diversity
 * from one generation of a population of genetic algorithm chromosomes to the
 * next. It is analogous to biological mutation.
 */

/**
 * @brief Enumeration of supported mutation strategies.
 */
typedef enum {
    UNIFORM_MUTATION,     /**< Adds a random value from a uniform distribution to the gene. */
    GAUSSIAN_MUTATION,    /**< Adds a random value from a Gaussian (normal) distribution to the gene. */
    NON_UNIFORM_MUTATION, /**< A variant of uniform mutation where the mutation rate decreases over generations. */
    ADAPTIVE_MUTATION     /**< Adjusts the mutation rate based on the population's fitness diversity. */
} MutationType;


/**
 * @brief Mutates a neural network's weights and biases in-place.
 * @details This function iterates through the weights and biases of the given
 * network. For each parameter, it decides whether to apply a mutation based on
 * `mutation_chance`. If a mutation occurs, the type and magnitude are determined
 * by the other parameters. The network's parameters are modified directly.
 * @param network The neural network to mutate. This network is modified directly.
 * @param mutation_rate The base magnitude of the mutation. Its interpretation depends on the mutation type.
 * @param mutation_chance The probability (0.0 to 1.0) that any single weight or bias will be mutated.
 * @param mutation_type The type of mutation to perform (e.g., `GAUSSIAN_MUTATION`).
 * @param mutation_std_dev The standard deviation for `GAUSSIAN_MUTATION`.
 * @param current_gen The current generation number, used by `NON_UNIFORM_MUTATION`.
 * @param max_gens The maximum number of generations, used by `NON_UNIFORM_MUTATION`.
 * @param fitness_std_dev The standard deviation of the population's fitness, used by `ADAPTIVE_MUTATION`.
 */
void mutate_network(NeuralNetwork* network, float mutation_rate, float mutation_chance, MutationType mutation_type, double mutation_std_dev, int current_gen, int max_gens, double fitness_std_dev);

#endif // MUTATION_H
