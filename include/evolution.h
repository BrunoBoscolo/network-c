#ifndef EVOLUTION_H
#define EVOLUTION_H

#include "neural_network.h"
#include "crossover.h"

// A struct to hold a network and its fitness score
typedef struct {
    NeuralNetwork* network;
    double fitness;
} NetworkFitness;

// --- Evolution Functions ---

/**
 * @brief Creates the initial population of random neural networks.
 * @param population_size The number of networks in the population.
 * @param num_layers The number of layers in each network.
 * @param architecture The architecture of each network.
 * @param activation_hidden The activation function for the hidden layers.
 * @param activation_output The activation function for the output layer.
 * @return An array of pointers to the newly created networks. The caller is responsible
 * for freeing both the outer array and each `NeuralNetwork*` inside it.
 */
NeuralNetwork** evo_create_initial_population(int population_size, int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output);

/**
 * @brief Creates a new generation of networks through reproduction.
 * @param fittest_networks An array of the fittest networks from the previous generation.
 * @param num_fittest The number of fittest networks.
 * @param new_population_size The size of the new population to create.
 * @param crossover_type The crossover strategy to use.
 * @param tournament_size The size of the tournament for parent selection.
 * @return An array of pointers to the new generation of networks. The caller is responsible
 * for freeing both the outer array and each `NeuralNetwork*` inside it.
 */
NeuralNetwork** evo_reproduce(const NetworkFitness* fittest_networks, int num_fittest, int new_population_size, CrossoverType crossover_type, int tournament_size);

#endif // EVOLUTION_H
