#include "gann.h"
#include "selection.h"
#include "crossover.h"
#include "mutation.h"
#include "gann_errors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// --- Helper functions (private to this file) ---

// qsort comparison function for sorting networks by fitness in descending order
static int compare_fitness_desc(const void* a, const void* b) {
    const NetworkFitness* nf_a = (const NetworkFitness*)a;
    const NetworkFitness* nf_b = (const NetworkFitness*)b;
    if (nf_a->fitness < nf_b->fitness) return 1;
    if (nf_a->fitness > nf_b->fitness) return -1;
    return 0;
}

// Helper to get the index of the max value in a matrix row (the prediction)
static int get_predicted_class(const Matrix* output) {
    if (!output || !output->data || output->cols == 0) {
        gann_set_error(GANN_ERROR_INVALID_PARAM);
        return -1;
    }
    int max_index = 0;
    for (int i = 1; i < output->cols; i++) {
        if (output->data[0][i] > output->data[0][max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

// Helper to get the true class from a one-hot encoded label vector
static int get_true_class(const double* label_row, int num_classes) {
    if (!label_row) return -1;
    for (int i = 0; i < num_classes; i++) {
        if (label_row[i] == 1.0) {
            return i;
        }
    }
    return -1; // Should not happen with valid data
}


// Fitness function used by the training loop
static double calculate_fitness(NeuralNetwork* network, const Dataset* dataset, int num_samples) {
    int correct_predictions = 0;
    if (num_samples <= 0 || num_samples > dataset->num_items) {
        num_samples = dataset->num_items;
    }

    // Create a single input matrix to be reused to avoid repeated malloc/free calls.
    Matrix* input = create_matrix(1, dataset->images->cols);
    if (!input) {
        // create_matrix sets the error, but this is a private helper.
        // We don't propagate the error code here, just return 0 fitness.
        return 0.0;
    }

    for (int i = 0; i < num_samples; i++) {
        // Copy the current sample's data into the reusable input matrix.
        memcpy(input->data[0], dataset->images->data[i], dataset->images->cols * sizeof(double));

        Matrix* output = nn_forward_pass(network, input);
        if (!output) {
            // nn_forward_pass sets the error, so we can just skip.
            continue;
        }

        int predicted_class = get_predicted_class(output);
        int num_classes = network->architecture[network->num_layers - 1];
        int true_class = get_true_class(dataset->labels->data[i], num_classes);

        if (predicted_class == true_class) {
            correct_predictions++;
        }

        free_matrix(output); // Output matrix is newly created in each pass, so it must be freed.
    }

    free_matrix(input); // Free the reusable input matrix.
    return (double)correct_predictions / num_samples;
}


// --- High-Level API Implementation ---

void gann_seed_rng(unsigned int seed) {
    srand(seed);
    gann_set_error(GANN_SUCCESS);
}

GannTrainParams gann_create_default_params(void) {
    GannTrainParams params = {
        .architecture = NULL,
        .num_layers = 0,
        .population_size = 50,
        .num_generations = 100,
        .mutation_rate = 0.1f,
        .mutation_chance = 0.25f,
        .fitness_samples = 1000,
        .selection_type = TOURNAMENT_SELECTION,
        .tournament_size = 5,
        .elitism_count = 1,
        .activation_hidden = RELU,
        .activation_output = SIGMOID,
        .crossover_type = UNIFORM_CROSSOVER,
        .mutation_type = GAUSSIAN_MUTATION,
        .mutation_std_dev = 0.1,
        .logging = true
    };
    gann_set_error(GANN_SUCCESS);
    return params;
}

NeuralNetwork* gann_evolve(const GannEvolveParams* params, const Dataset* train_dataset) {
    const GannTrainParams* base_params = &params->base_params;

    if (!base_params || !train_dataset || !base_params->architecture) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    if (base_params->num_layers < 2 || base_params->population_size <= 0 || base_params->num_generations <= 0 ||
        base_params->mutation_rate < 0.0f || base_params->mutation_chance < 0.0f || base_params->mutation_chance > 1.0f) {
        gann_set_error(GANN_ERROR_INVALID_PARAM);
        return NULL;
    }

    // --- 1. Create Initial Population ---
    NeuralNetwork** population = evo_create_initial_population(base_params->population_size, base_params->num_layers, base_params->architecture, base_params->activation_hidden, base_params->activation_output);
    if (!population) {
        // evo_create_initial_population should set the error.
        return NULL;
    }

    if (base_params->logging) {
        printf("Created initial population of %d networks.\n", base_params->population_size);
        printf("Starting evolution for %d generations...\n", base_params->num_generations);
    }

    // --- 2. Run Evolutionary Loop ---
    for (int gen = 0; gen < base_params->num_generations; gen++) {
        NetworkFitness* population_with_fitness = malloc(base_params->population_size * sizeof(NetworkFitness));
        if (!population_with_fitness) {
             gann_set_error(GANN_ERROR_ALLOC_FAILED);
             break; // Exit loop
        }

        double best_accuracy_in_gen = 0.0;
        double fitness_sum = 0;
        for (int i = 0; i < base_params->population_size; i++) {
            population_with_fitness[i].network = population[i];
            population_with_fitness[i].fitness = calculate_fitness(population[i], train_dataset, base_params->fitness_samples);
            fitness_sum += population_with_fitness[i].fitness;
            if (population_with_fitness[i].fitness > best_accuracy_in_gen) {
                best_accuracy_in_gen = population_with_fitness[i].fitness;
            }
        }

        double fitness_mean = fitness_sum / base_params->population_size;
        double fitness_std_dev = 0;
        for (int i = 0; i < base_params->population_size; i++) {
            fitness_std_dev += pow(population_with_fitness[i].fitness - fitness_mean, 2);
        }
        fitness_std_dev = sqrt(fitness_std_dev / base_params->population_size);

        if (base_params->logging) {
            printf("Generation %d/%d | Best Accuracy: %.2f%% | Avg Fitness: %.4f | Fitness StdDev: %.4f\n",
                   gen + 1, base_params->num_generations, best_accuracy_in_gen * 100.0, fitness_mean, fitness_std_dev);
        }

        int num_fittest;
        NetworkFitness* fittest_networks_info = params->selection_func(population_with_fitness, base_params->population_size, &num_fittest, (SelectionType)base_params->selection_type, base_params->tournament_size);

        // --- Elitism: Preserve the best networks ---
        int elitism_count = base_params->elitism_count;
        if (elitism_count > base_params->population_size) elitism_count = base_params->population_size;

        NeuralNetwork** elite_networks = NULL;
        if (elitism_count > 0) {
            qsort(population_with_fitness, base_params->population_size, sizeof(NetworkFitness), compare_fitness_desc);
            elite_networks = malloc(elitism_count * sizeof(NeuralNetwork*));
            if(elite_networks != NULL) {
                for (int i = 0; i < elitism_count; i++) {
                    elite_networks[i] = nn_clone(population_with_fitness[i].network);
                }
            }
        }

        // --- Reproduction ---
        int children_to_create = base_params->population_size - elitism_count;
        NeuralNetwork** new_population = evo_reproduce(fittest_networks_info, num_fittest, children_to_create, (CrossoverType)base_params->crossover_type, base_params->tournament_size);
        if (new_population == NULL) {
            // Handle reproduction failure
            free(population_with_fitness);
            free(fittest_networks_info);
            if (elite_networks) {
                for(int i=0; i<elitism_count; i++) nn_free(elite_networks[i]);
                free(elite_networks);
            }
            break;
        }

        // Mutate the new children
        for (int i = 0; i < children_to_create; i++) {
            params->mutation_func(new_population[i], base_params->mutation_rate, base_params->mutation_chance, (MutationType)base_params->mutation_type, base_params->mutation_std_dev, gen, base_params->num_generations, fitness_std_dev);
        }

        // --- Combine elites and children ---
        if (elitism_count > 0 && elite_networks != NULL) {
            // The `new_population` array is currently of size `children_to_create`.
            // We need to resize it to fit the elite networks as well.
            NeuralNetwork** final_population = realloc(new_population, base_params->population_size * sizeof(NeuralNetwork*));
            if (final_population) {
                new_population = final_population;
                // Copy elite networks into the final population array
                for (int i = 0; i < elitism_count; i++) {
                    new_population[children_to_create + i] = elite_networks[i];
                }
            }
            free(elite_networks); // free the container for clones
        }


        // Free the old population's networks before replacing the population
        for (int i = 0; i < base_params->population_size; i++) {
            nn_free(population[i]);
        }
        free(population); // Free the array of pointers
        free(population_with_fitness);
        free(fittest_networks_info);

        population = new_population; // Point to the new generation
    }

    // --- 3. Find Best Network from the final population ---
    NeuralNetwork* best_net = NULL;
    double best_overall_accuracy = 0.0;
    for (int i = 0; i < base_params->population_size; i++) {
        double accuracy = calculate_fitness(population[i], train_dataset, train_dataset->num_items); // Final evaluation on full dataset
        if (accuracy > best_overall_accuracy) {
            best_overall_accuracy = accuracy;
            // We need to clone the best network, because the population will be freed.
            if (best_net) nn_free(best_net);
            best_net = nn_clone(population[i]);
            if (!best_net) {
                // nn_clone failed and set the error. We can't continue.
                break;
            }
        }
    }

    if (base_params->logging) {
        printf("Evolution finished. Best accuracy: %.2f%%\n", best_overall_accuracy * 100.0);
    }

    // --- 4. Cleanup ---
    for (int i = 0; i < base_params->population_size; i++) {
        nn_free(population[i]);
    }
    free(population);

    if (best_net) {
        gann_set_error(GANN_SUCCESS);
    }
    // If best_net is NULL, an error has already been set.
    return best_net;
}

NeuralNetwork* gann_train(const GannTrainParams* params, const Dataset* train_dataset) {
    // Add defensive checks at the beginning of the public API function.
    if (params == NULL || train_dataset == NULL || params->architecture == NULL) {
        fprintf(stderr, "Error: Cannot train network. Provided params, dataset or architecture is NULL.\n");
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    GannEvolveParams evolve_params = {
        .base_params = *params,
        .selection_func = select_fittest,
        .crossover_func = crossover,
        .mutation_func = mutate_network
    };
    return gann_evolve(&evolve_params, train_dataset);
}

int gann_predict(const NeuralNetwork* net, const double* input_data) {
    if (!net || !input_data) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return -1; // Invalid input
    }

    // Create a matrix for the input data
    Matrix* input_matrix = create_matrix(1, net->architecture[0]);
    if (!input_matrix) {
        // create_matrix sets the error
        return -1;
    }
    memcpy(input_matrix->data[0], input_data, net->architecture[0] * sizeof(double));

    // Perform the forward pass
    Matrix* output_matrix = nn_forward_pass(net, input_matrix);
    if (!output_matrix) {
        free_matrix(input_matrix);
        // nn_forward_pass sets the error
        return -1;
    }

    // Get the result
    int prediction = get_predicted_class(output_matrix);

    // Cleanup
    free_matrix(input_matrix);
    free_matrix(output_matrix);

    gann_set_error(GANN_SUCCESS);
    return prediction;
}

double gann_evaluate(const NeuralNetwork* net, const Dataset* dataset) {
    if (!net || !dataset) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return 0.0;
    }

    int correct_predictions = 0;
    for (int i = 0; i < dataset->num_items; i++) {
        int prediction = gann_predict(net, dataset->images->data[i]);
        if (prediction == -1) {
            // An error occurred in gann_predict, and it has set the error code.
            // We can't continue evaluating, so we return 0.0 accuracy.
            return 0.0;
        }
        int num_classes = net->architecture[net->num_layers - 1];
        int true_class = get_true_class(dataset->labels->data[i], num_classes);

        if (prediction == true_class) {
            correct_predictions++;
        }
    }

    gann_set_error(GANN_SUCCESS);
    return (double)correct_predictions / dataset->num_items;
}
