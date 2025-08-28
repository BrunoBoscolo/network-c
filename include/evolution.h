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

NeuralNetwork* crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2);
NeuralNetwork** create_initial_population(int population_size, int num_layers, const int* architecture);
NetworkFitness* select_fittest(NetworkFitness* population_with_fitness, int population_size, int* num_fittest, SelectionType selection_type, int tournament_size);
NeuralNetwork** reproduce(const NetworkFitness* fittest_networks, int num_fittest, int new_population_size, float mutation_rate, float mutation_chance);

#endif // EVOLUTION_H
