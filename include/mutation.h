#ifndef MUTATION_H
#define MUTATION_H

#include "neural_network.h"

// Enum for mutation strategies
typedef enum {
    UNIFORM_MUTATION,
    GAUSSIAN_MUTATION,
    NON_UNIFORM_MUTATION,
    ADAPTIVE_MUTATION
} MutationType;


/**
 * @brief Mutates a neural network's weights and biases.
 * @param network The neural network to mutate.
 * @param mutation_rate The magnitude of the mutation.
 * @param mutation_chance The chance of a mutation occurring.
 * @param mutation_type The type of mutation to perform.
 * @param mutation_std_dev The standard deviation for Gaussian mutation.
 * @param current_gen The current generation number.
 * @param max_gens The maximum number of generations.
 * @param fitness_std_dev The standard deviation of the fitness scores of the population.
 */
void mutate_network(NeuralNetwork* network, float mutation_rate, float mutation_chance, MutationType mutation_type, double mutation_std_dev, int current_gen, int max_gens, double fitness_std_dev);

#endif // MUTATION_H
