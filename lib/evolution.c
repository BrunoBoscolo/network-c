#include "evolution.h"
#include <stdlib.h>
#include <stdio.h>

// --- Evolution Functions Implementation ---

// Creates an initial population of neural networks
NeuralNetwork** create_initial_population(int population_size, int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output) {
    NeuralNetwork** population = (NeuralNetwork**)malloc(population_size * sizeof(NeuralNetwork*));
    if (!population) return NULL;

    for (int i = 0; i < population_size; i++) {
        population[i] = create_neural_network(num_layers, architecture, activation_hidden, activation_output);
        if (population[i]) {
            initialize_network(population[i]);
        }
        // TODO: Handle allocation failure
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

// Selects the top-performing networks (elitism)
NetworkFitness* select_fittest_elitism(NetworkFitness* population_with_fitness, int population_size, int* num_fittest) {
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

// Selects networks using a tournament
NetworkFitness* select_fittest_tournament(NetworkFitness* population_with_fitness, int population_size, int* num_fittest, int tournament_size) {
    *num_fittest = population_size / 2;
    NetworkFitness* fittest = (NetworkFitness*)malloc(*num_fittest * sizeof(NetworkFitness));
    if (!fittest) {
        *num_fittest = 0;
        return NULL;
    }

    for (int i = 0; i < *num_fittest; i++) {
        int best_index = -1;
        double best_fitness = -1.0;

        // Run a tournament
        for (int j = 0; j < tournament_size; j++) {
            int competitor_index = rand() % population_size;
            if (population_with_fitness[competitor_index].fitness > best_fitness) {
                best_fitness = population_with_fitness[competitor_index].fitness;
                best_index = competitor_index;
            }
        }
        fittest[i] = population_with_fitness[best_index];
    }

    return fittest;
}

// Wrapper function to select fittest based on strategy
NetworkFitness* select_fittest(NetworkFitness* population_with_fitness, int population_size, int* num_fittest, SelectionType selection_type, int tournament_size) {
    switch (selection_type) {
        case ELITE:
            return select_fittest_elitism(population_with_fitness, population_size, num_fittest);
        case TOURNAMENT:
            return select_fittest_tournament(population_with_fitness, population_size, num_fittest, tournament_size);
        default:
            // Default to elite selection
            return select_fittest_elitism(population_with_fitness, population_size, num_fittest);
    }
}

// Performs uniform crossover between two parent networks.
// For each weight and bias, the child's value is randomly taken from one of the two parents.
NeuralNetwork* uniform_crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2) {
    if (!parent1 || !parent2 || parent1->num_layers != parent2->num_layers) {
        return NULL;
    }

    // Create a new network with the same architecture
    NeuralNetwork* child = create_neural_network(parent1->num_layers, parent1->architecture, parent1->activation_hidden, parent1->activation_output);
    if (!child) return NULL;

    // Perform uniform crossover for weights and biases
    for (int i = 0; i < parent1->num_layers - 1; i++) {
        // Weights
        for (int r = 0; r < parent1->weights[i]->rows; r++) {
            for (int c = 0; c < parent1->weights[i]->cols; c++) {
                if ((double)rand() / RAND_MAX > 0.5) {
                    child->weights[i]->data[r][c] = parent1->weights[i]->data[r][c];
                } else {
                    child->weights[i]->data[r][c] = parent2->weights[i]->data[r][c];
                }
            }
        }
        // Biases
        for (int c = 0; c < parent1->biases[i]->cols; c++) {
            if ((double)rand() / RAND_MAX > 0.5) {
                child->biases[i]->data[0][c] = parent1->biases[i]->data[0][c];
            } else {
                child->biases[i]->data[0][c] = parent2->biases[i]->data[0][c];
            }
        }
    }

    return child;
}

// Performs single-point crossover between two parent networks.
NeuralNetwork* single_point_crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2) {
    if (!parent1 || !parent2 || parent1->num_layers != parent2->num_layers) {
        return NULL;
    }

    NeuralNetwork* child = create_neural_network(parent1->num_layers, parent1->architecture, parent1->activation_hidden, parent1->activation_output);
    if (!child) return NULL;

    int total_weights = 0;
    for (int i = 0; i < parent1->num_layers - 1; i++) {
        total_weights += parent1->weights[i]->rows * parent1->weights[i]->cols;
        total_weights += parent1->biases[i]->cols;
    }

    int crossover_point = rand() % total_weights;
    int current_weight = 0;

    for (int i = 0; i < parent1->num_layers - 1; i++) {
        // Weights
        for (int r = 0; r < parent1->weights[i]->rows; r++) {
            for (int c = 0; c < parent1->weights[i]->cols; c++) {
                if (current_weight < crossover_point) {
                    child->weights[i]->data[r][c] = parent1->weights[i]->data[r][c];
                } else {
                    child->weights[i]->data[r][c] = parent2->weights[i]->data[r][c];
                }
                current_weight++;
            }
        }
        // Biases
        for (int c = 0; c < parent1->biases[i]->cols; c++) {
            if (current_weight < crossover_point) {
                child->biases[i]->data[0][c] = parent1->biases[i]->data[0][c];
            } else {
                child->biases[i]->data[0][c] = parent2->biases[i]->data[0][c];
            }
            current_weight++;
        }
    }

    return child;
}

// Performs two-point crossover between two parent networks.
NeuralNetwork* two_point_crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2) {
    if (!parent1 || !parent2 || parent1->num_layers != parent2->num_layers) {
        return NULL;
    }

    NeuralNetwork* child = create_neural_network(parent1->num_layers, parent1->architecture, parent1->activation_hidden, parent1->activation_output);
    if (!child) return NULL;

    int total_weights = 0;
    for (int i = 0; i < parent1->num_layers - 1; i++) {
        total_weights += parent1->weights[i]->rows * parent1->weights[i]->cols;
        total_weights += parent1->biases[i]->cols;
    }

    int crossover_point1 = rand() % total_weights;
    int crossover_point2 = rand() % total_weights;
    if (crossover_point1 > crossover_point2) {
        int temp = crossover_point1;
        crossover_point1 = crossover_point2;
        crossover_point2 = temp;
    }

    int current_weight = 0;

    for (int i = 0; i < parent1->num_layers - 1; i++) {
        // Weights
        for (int r = 0; r < parent1->weights[i]->rows; r++) {
            for (int c = 0; c < parent1->weights[i]->cols; c++) {
                if (current_weight >= crossover_point1 && current_weight < crossover_point2) {
                    child->weights[i]->data[r][c] = parent2->weights[i]->data[r][c];
                } else {
                    child->weights[i]->data[r][c] = parent1->weights[i]->data[r][c];
                }
                current_weight++;
            }
        }
        // Biases
        for (int c = 0; c < parent1->biases[i]->cols; c++) {
            if (current_weight >= crossover_point1 && current_weight < crossover_point2) {
                child->biases[i]->data[0][c] = parent2->biases[i]->data[0][c];
            } else {
                child->biases[i]->data[0][c] = parent1->biases[i]->data[0][c];
            }
            current_weight++;
        }
    }

    return child;
}

NeuralNetwork* crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2, CrossoverType crossover_type) {
    switch (crossover_type) {
        case UNIFORM:
            return uniform_crossover(parent1, parent2);
        case SINGLE_POINT:
            return single_point_crossover(parent1, parent2);
        case TWO_POINT:
            return two_point_crossover(parent1, parent2);
        default:
            return uniform_crossover(parent1, parent2);
    }
}

// Creates a new generation using crossover
NeuralNetwork** reproduce(const NetworkFitness* fittest_networks, int num_fittest, int new_population_size, CrossoverType crossover_type) {
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
        NeuralNetwork* child = crossover(parent1, parent2, crossover_type);
        if (!child) {
            // Handle crossover failure, e.g., by cloning one parent
            child = clone_network(parent1);
        }

        new_population[i] = child;
    }

    return new_population;
}
