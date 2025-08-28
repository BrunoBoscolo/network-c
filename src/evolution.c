#include "evolution.h"
#include <stdlib.h>
#include <stdio.h>

// --- Evolution Functions Implementation ---

// Creates an initial population of neural networks
NeuralNetwork** create_initial_population(int population_size, int num_layers, const int* architecture) {
    NeuralNetwork** population = (NeuralNetwork**)malloc(population_size * sizeof(NeuralNetwork*));
    if (!population) return NULL;

    for (int i = 0; i < population_size; i++) {
        population[i] = create_neural_network(num_layers, architecture);
    }
    return population;
}

// Comparison function for qsort to sort networks by fitness in descending order
int compare_fitness(const void* a, const void* b) {
    const NetworkFitness* nf_a = (const NetworkFitness*)a;
    const NetworkFitness* nf_b = (const NetworkFitness*)b;
    if (nf_a->fitness < nf_b->fitness) return 1;
    if (nf_a->fitness > nf_b->fitness) return -1;
    return 0;
}

// Selects the fittest networks from a population
NetworkFitness* select_fittest(NetworkFitness* population_with_fitness, int population_size, int* num_fittest) {
    // Sort the population by fitness
    qsort(population_with_fitness, population_size, sizeof(NetworkFitness), compare_fitness);

    // Select the top half
    *num_fittest = population_size / 2;
    NetworkFitness* fittest = (NetworkFitness*)malloc(*num_fittest * sizeof(NetworkFitness));
    if (!fittest) {
        *num_fittest = 0;
        return NULL;
    }

    for (int i = 0; i < *num_fittest; i++) {
        fittest[i] = population_with_fitness[i];
    }

    return fittest;
}

// Performs crossover between two parent networks to produce a child.
// The child's weights and biases are the average of the parents'.
NeuralNetwork* crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2) {
    if (!parent1 || !parent2 || parent1->num_layers != parent2->num_layers) {
        return NULL;
    }

    // Create a new network with the same architecture
    NeuralNetwork* child = create_neural_network(parent1->num_layers, parent1->architecture);
    if (!child) return NULL;

    // Average the weights and biases
    for (int i = 0; i < parent1->num_layers - 1; i++) {
        // Weights
        for (int r = 0; r < parent1->weights[i]->rows; r++) {
            for (int c = 0; c < parent1->weights[i]->cols; c++) {
                child->weights[i]->data[r][c] = (parent1->weights[i]->data[r][c] + parent2->weights[i]->data[r][c]) / 2.0;
            }
        }
        // Biases
        for (int c = 0; c < parent1->biases[i]->cols; c++) {
            child->biases[i]->data[0][c] = (parent1->biases[i]->data[0][c] + parent2->biases[i]->data[0][c]) / 2.0;
        }
    }

    return child;
}

// Creates a new generation using crossover and mutation
NeuralNetwork** reproduce(const NetworkFitness* fittest_networks, int num_fittest, int new_population_size, float mutation_rate, float mutation_chance) {
    if (num_fittest == 0) return NULL;

    NeuralNetwork** new_population = (NeuralNetwork**)malloc(new_population_size * sizeof(NeuralNetwork*));
    if (!new_population) return NULL;

    for (int i = 0; i < new_population_size; i++) {
        // Choose two random parents from the fittest networks
        int parent1_index = rand() % num_fittest;
        int parent2_index = rand() % num_fittest;
        const NeuralNetwork* parent1 = fittest_networks[parent1_index].network;
        const NeuralNetwork* parent2 = fittest_networks[parent2_index].network;

        // Create a child using crossover
        NeuralNetwork* child = crossover(parent1, parent2);
        if (!child) {
            // Handle crossover failure, e.g., by cloning one parent
            child = clone_network(parent1);
        }

        // Mutate the child
        mutate_network(child, mutation_rate, mutation_chance);

        new_population[i] = child;
    }

    return new_population;
}
