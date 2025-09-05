#ifndef CROSSOVER_H
#define CROSSOVER_H

#include "neural_network.h"

// Enum for crossover strategies
typedef enum {
    UNIFORM_CROSSOVER,
    SINGLE_POINT_CROSSOVER,
    TWO_POINT_CROSSOVER,
    ARITHMETIC_CROSSOVER
} CrossoverType;


/**
 * @brief Performs crossover between two parent networks to create a child.
 * @param parent1 The first parent network.
 * @param parent2 The second parent network.
 * @param crossover_type The crossover strategy to use.
 * @return A new network created by combining the parents' genes.
 */
NeuralNetwork* crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2, CrossoverType crossover_type);

#endif // CROSSOVER_H
