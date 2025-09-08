#include "selection.h"
#include <stdlib.h>
#include <stdio.h>

// Comparison function for qsort to sort networks by fitness in descending order
static int compare_fitness(const void* a, const void* b) {
    const NetworkFitness* nf_a = (const NetworkFitness*)a;
    const NetworkFitness* nf_b = (const NetworkFitness*)b;
    if (nf_a->fitness < nf_b->fitness) return 1;
    if (nf_a->fitness > nf_b->fitness) return -1;
    return 0;
}

// Selects the top-performing networks (elitism)
static NetworkFitness* select_fittest_elitism(NetworkFitness* population_with_fitness, int population_size, int* num_fittest) {
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

// Selects networks using rank selection
static NetworkFitness* select_fittest_rank(NetworkFitness* population_with_fitness, int population_size, int* num_fittest) {
    *num_fittest = population_size / 2;
    NetworkFitness* fittest = (NetworkFitness*)malloc(*num_fittest * sizeof(NetworkFitness));
    if (!fittest) {
        *num_fittest = 0;
        return NULL;
    }

    // Sort the population by fitness
    qsort(population_with_fitness, population_size, sizeof(NetworkFitness), compare_fitness);

    // Calculate total rank sum
    double total_rank_sum = 0;
    for (int i = 0; i < population_size; i++) {
        total_rank_sum += (population_size - i);
    }

    for (int i = 0; i < *num_fittest; i++) {
        double slice = (double)rand() / RAND_MAX * total_rank_sum;
        double current_rank_sum = 0;
        for (int j = 0; j < population_size; j++) {
            current_rank_sum += (population_size - j);
            if (current_rank_sum >= slice) {
                fittest[i] = population_with_fitness[j];
                break;
            }
        }
    }

    return fittest;
}

// Selects networks using a tournament
static NetworkFitness* select_fittest_tournament(NetworkFitness* population_with_fitness, int population_size, int* num_fittest, int tournament_size) {
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

// Selects networks using roulette wheel selection
static NetworkFitness* select_fittest_roulette_wheel(NetworkFitness* population_with_fitness, int population_size, int* num_fittest) {
    *num_fittest = population_size / 2;
    NetworkFitness* fittest = (NetworkFitness*)malloc(*num_fittest * sizeof(NetworkFitness));
    if (!fittest) {
        *num_fittest = 0;
        return NULL;
    }

    double total_fitness = 0;
    for (int i = 0; i < population_size; i++) {
        total_fitness += population_with_fitness[i].fitness;
    }

    for (int i = 0; i < *num_fittest; i++) {
        double slice = (double)rand() / RAND_MAX * total_fitness;
        double current_fitness = 0;
        for (int j = 0; j < population_size; j++) {
            current_fitness += population_with_fitness[j].fitness;
            if (current_fitness >= slice) {
                fittest[i] = population_with_fitness[j];
                break;
            }
        }
    }

    return fittest;
}


// Wrapper function to select fittest based on strategy
NetworkFitness* select_fittest(NetworkFitness* population_with_fitness, int population_size, int* num_fittest, SelectionType selection_type, int tournament_size) {
    if (population_with_fitness == NULL || num_fittest == NULL) {
        fprintf(stderr, "Error: Cannot select fittest. Provided population or num_fittest pointer is NULL.\n");
        if (num_fittest) *num_fittest = 0;
        return NULL;
    }
    switch (selection_type) {
        case ELITISM_SELECTION:
            return select_fittest_elitism(population_with_fitness, population_size, num_fittest);
        case TOURNAMENT_SELECTION:
            return select_fittest_tournament(population_with_fitness, population_size, num_fittest, tournament_size);
        case ROULETTE_WHEEL_SELECTION:
            return select_fittest_roulette_wheel(population_with_fitness, population_size, num_fittest);
        case RANK_SELECTION:
            return select_fittest_rank(population_with_fitness, population_size, num_fittest);
        default:
            // Default to elite selection
            return select_fittest_elitism(population_with_fitness, population_size, num_fittest);
    }
}
