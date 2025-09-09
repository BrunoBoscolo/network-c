#ifndef CROSSOVER_H
#define CROSSOVER_H

#include "neural_network.h"

/**
 * @file crossover.h
 * @brief Defines strategies for combining parent networks to create offspring.
 * @details Crossover (or recombination) is the process of taking two parent
 * solutions and producing a child solution from them. This is a key part of
 * how a genetic algorithm explores the solution space.
 */

/**
 * @brief Enumeration of supported crossover strategies.
 */
typedef enum {
    UNIFORM_CROSSOVER,      /**< Each gene (weight/bias) is chosen from either parent with equal probability. */
    SINGLE_POINT_CROSSOVER, /**< A single point is chosen, and genes are swapped between parents after this point. */
    TWO_POINT_CROSSOVER,    /**< Two points are chosen, and genes between the points are swapped. */
    ARITHMETIC_CROSSOVER    /**< The child's genes are a weighted average of the parents' genes. */
} CrossoverType;


/**
 * @brief Performs crossover between two parent networks to create a child network.
 * @details This function acts as a dispatcher, calling the appropriate underlying
 * crossover method based on the `crossover_type` parameter. The child network
 * inherits its structure from the parents but gets a new set of weights and
 * biases that are a combination of the parents' parameters.
 * @param parent1 A pointer to the first parent network.
 * @param parent2 A pointer to the second parent network.
 * @param crossover_type The crossover strategy to use (e.g., `UNIFORM_CROSSOVER`).
 * @return A new `NeuralNetwork` representing the child. The caller is
 *         responsible for freeing this new network using `nn_free()`.
 * @return `NULL` on failure (e.g., if parents are incompatible).
 */
NeuralNetwork* crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2, CrossoverType crossover_type);

#endif // CROSSOVER_H
