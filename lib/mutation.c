#include "mutation.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Helper function for generating a random number from a Gaussian distribution
static double randn(double mu, double sigma) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mu + sigma * z;
}

// Simple uniform mutation
static void uniform_mutation(NeuralNetwork* network, float mutation_rate, float mutation_chance) {
    for (int i = 0; i < network->num_layers - 1; i++) {
        // Mutate weights
        for (int r = 0; r < network->weights[i]->rows; r++) {
            for (int c = 0; c < network->weights[i]->cols; c++) {
                if ((double)rand() / RAND_MAX < mutation_chance) {
                    double mutation = ((double)rand() / RAND_MAX - 0.5) * 2.0 * mutation_rate;
                    network->weights[i]->data[r][c] += mutation;
                }
            }
        }
        // Mutate biases
        for (int c = 0; c < network->biases[i]->cols; c++) {
            if ((double)rand() / RAND_MAX < mutation_chance) {
                double mutation = ((double)rand() / RAND_MAX - 0.5) * 2.0 * mutation_rate;
                network->biases[i]->data[0][c] += mutation;
            }
        }
    }
}

// Gaussian mutation
static void gaussian_mutation(NeuralNetwork* network, float mutation_chance, double std_dev) {
    for (int i = 0; i < network->num_layers - 1; i++) {
        // Mutate weights
        for (int r = 0; r < network->weights[i]->rows; r++) {
            for (int c = 0; c < network->weights[i]->cols; c++) {
                if ((double)rand() / RAND_MAX < mutation_chance) {
                    network->weights[i]->data[r][c] += randn(0, std_dev);
                }
            }
        }
        // Mutate biases
        for (int c = 0; c < network->biases[i]->cols; c++) {
            if ((double)rand() / RAND_MAX < mutation_chance) {
                network->biases[i]->data[0][c] += randn(0, std_dev);
            }
        }
    }
}


// Non-uniform mutation
static void non_uniform_mutation(NeuralNetwork* network, float mutation_rate, float mutation_chance, int current_gen, int max_gens) {
    float current_mutation_rate = mutation_rate * (1.0 - (double)current_gen / max_gens);
    for (int i = 0; i < network->num_layers - 1; i++) {
        // Mutate weights
        for (int r = 0; r < network->weights[i]->rows; r++) {
            for (int c = 0; c < network->weights[i]->cols; c++) {
                if ((double)rand() / RAND_MAX < mutation_chance) {
                    double mutation = ((double)rand() / RAND_MAX - 0.5) * 2.0 * current_mutation_rate;
                    network->weights[i]->data[r][c] += mutation;
                }
            }
        }
        // Mutate biases
        for (int c = 0; c < network->biases[i]->cols; c++) {
            if ((double)rand() / RAND_MAX < mutation_chance) {
                double mutation = ((double)rand() / RAND_MAX - 0.5) * 2.0 * current_mutation_rate;
                network->biases[i]->data[0][c] += mutation;
            }
        }
    }
}


// Adaptive mutation
static void adaptive_mutation(NeuralNetwork* network, float initial_mutation_rate, float mutation_chance, double fitness_std_dev) {
    // Adjust mutation rate based on fitness diversity.
    // This is a simple example. A more sophisticated approach could be used.
    float mutation_rate = initial_mutation_rate;
    if (fitness_std_dev < 0.05) { // Low diversity
        mutation_rate *= 1.5;
    } else if (fitness_std_dev > 0.2) { // High diversity
        mutation_rate *= 0.75;
    }

    for (int i = 0; i < network->num_layers - 1; i++) {
        // Mutate weights
        for (int r = 0; r < network->weights[i]->rows; r++) {
            for (int c = 0; c < network->weights[i]->cols; c++) {
                if ((double)rand() / RAND_MAX < mutation_chance) {
                    double mutation = ((double)rand() / RAND_MAX - 0.5) * 2.0 * mutation_rate;
                    network->weights[i]->data[r][c] += mutation;
                }
            }
        }
        // Mutate biases
        for (int c = 0; c < network->biases[i]->cols; c++) {
            if ((double)rand() / RAND_MAX < mutation_chance) {
                double mutation = ((double)rand() / RAND_MAX - 0.5) * 2.0 * mutation_rate;
                network->biases[i]->data[0][c] += mutation;
            }
        }
    }
}


void mutate_network(NeuralNetwork* network, float mutation_rate, float mutation_chance, MutationType mutation_type, double mutation_std_dev, int current_gen, int max_gens, double fitness_std_dev) {
    if (network == NULL) {
        fprintf(stderr, "Error: Cannot mutate network. Provided network is NULL.\n");
        return;
    }
    switch (mutation_type) {
        case UNIFORM_MUTATION:
            uniform_mutation(network, mutation_rate, mutation_chance);
            break;
        case GAUSSIAN_MUTATION:
            gaussian_mutation(network, mutation_chance, mutation_std_dev);
            break;
        case NON_UNIFORM_MUTATION:
            non_uniform_mutation(network, mutation_rate, mutation_chance, current_gen, max_gens);
            break;
        case ADAPTIVE_MUTATION:
            adaptive_mutation(network, mutation_rate, mutation_chance, fitness_std_dev);
            break;
        default:
            uniform_mutation(network, mutation_rate, mutation_chance);
            break;
    }
}
