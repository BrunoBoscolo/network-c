#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "neural_network.h"

// Enum for selection strategies
typedef enum {
    ELITE,
    TOURNAMENT
} SelectionType;

// A struct to hold a network and its fitness score
typedef struct {
    NeuralNetwork* network;
    double fitness;
} NetworkFitness;

// --- Evolution Functions ---

/**
 * @brief Performs crossover between two parent networks to create a child.
 * @param parent1 The first parent network.
 * @param parent2 The second parent network.
 * @return A new network created by combining the parents' genes.
 */
NeuralNetwork* crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2);

/**
 * @brief Creates the initial population of random neural networks.
 * @param population_size The number of networks in the population.
 * @param num_layers The number of layers in each network.
 * @param architecture The architecture of each network.
 * @param activation_hidden The activation function for the hidden layers.
 * @param activation_output The activation function for the output layer.
 * @return An array of pointers to the newly created networks.
 */
NeuralNetwork** create_initial_population(int population_size, int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output);

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

/**
 * @brief Creates a new generation of networks through reproduction.
 * @param fittest_networks An array of the fittest networks from the previous generation.
 * @param num_fittest The number of fittest networks.
 * @param new_population_size The size of the new population to create.
 * @param mutation_rate The mutation rate.
 * @param mutation_chance The mutation chance.
 * @return An array of pointers to the new generation of networks.
 */
NeuralNetwork** reproduce(const NetworkFitness* fittest_networks, int num_fittest, int new_population_size, float mutation_rate, float mutation_chance);

#endif // EVOLUTION_H
