#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "neural_network.h"
#include "crossover.h"

/**
 * @file evolution.h
 * @brief Functions for the evolutionary aspects of the genetic algorithm.
 * @details This file contains the core logic for creating and evolving a
 * population of neural networks, including initialization and reproduction.
 */

/**
 * @brief A struct to associate a neural network with its calculated fitness score.
 * @details This is a convenience struct used during the evolution process to
 * keep track of how well each network in the population performs.
 */
typedef struct {
    NeuralNetwork* network; /**< A pointer to the neural network. */
    double fitness;         /**< The fitness score of the network (e.g., accuracy). */
} NetworkFitness;

// --- Evolution Functions ---

/**
 * @brief Creates the initial population of random neural networks.
 * @details Each network in the population is created with the specified
 * architecture and activation functions, and then its weights are initialized
 * with random values using `nn_init()`.
 * @param population_size The number of neural networks to create in the population.
 * @param num_layers The number of layers for each network.
 * @param architecture The architecture (number of neurons per layer) for each network.
 * @param activation_hidden The activation function to use for the hidden layers of each network.
 * @param activation_output The activation function to use for the output layer of each network.
 * @return An array of pointers to the newly created neural networks.
 * @return `NULL` on failure. The caller is responsible for freeing both the
 * returned array and each `NeuralNetwork*` within it.
 */
NeuralNetwork** evo_create_initial_population(int population_size, int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output);

/**
 * @brief Creates a new generation of networks through selection and crossover.
 * @details This function generates a new population of "child" networks from a
 * pool of "parent" networks. Parents are selected from the `fittest_networks`
 * pool using tournament selection. Two parents are chosen to create one child
 * via a specified crossover method.
 * @param fittest_networks An array of `NetworkFitness` structs representing the parent pool.
 * @param num_fittest The number of networks in the `fittest_networks` array.
 * @param new_population_size The desired number of child networks to create for the new generation.
 * @param crossover_type The crossover strategy (e.g., `UNIFORM_CROSSOVER`) to use for creating children.
 * @param tournament_size The number of individuals to compete in each parent selection tournament.
 * @return An array of pointers to the new generation of child networks.
 * @return `NULL` on failure. The caller is responsible for freeing both the
 * returned array and each `NeuralNetwork*` within it.
 */
NeuralNetwork** evo_reproduce(const NetworkFitness* fittest_networks, int num_fittest, int new_population_size, CrossoverType crossover_type, int tournament_size);

#endif // EVOLUTION_H
