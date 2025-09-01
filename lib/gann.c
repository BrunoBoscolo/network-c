#include "gann.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- Helper functions (private to this file) ---

// Helper to get the index of the max value in a matrix row (the prediction)
static int get_predicted_class(const Matrix* output) {
    int max_index = 0;
    for (int i = 1; i < output->cols; i++) {
        if (output->data[0][i] > output->data[0][max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

// Helper to get the true class from a one-hot encoded label vector
static int get_true_class(const double* label_row) {
    for (int i = 0; i < MNIST_NUM_CLASSES; i++) {
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

    for (int i = 0; i < num_samples; i++) {
        // Create a temporary matrix for a single image input
        Matrix* input = create_matrix(1, dataset->images->cols);
        if (!input) continue;
        memcpy(input->data[0], dataset->images->data[i], dataset->images->cols * sizeof(double));

        Matrix* output = forward_pass(network, input);
        if (!output) {
            free_matrix(input);
            continue;
        }

        int predicted_class = get_predicted_class(output);
        int true_class = get_true_class(dataset->labels->data[i]);

        if (predicted_class == true_class) {
            correct_predictions++;
        }

        free_matrix(input);
        free_matrix(output);
    }

    return (double)correct_predictions / num_samples;
}


// --- High-Level API Implementation ---

NeuralNetwork* gann_train(const GannTrainParams* params, const Dataset* train_dataset) {
    if (!params || !train_dataset) {
        return NULL;
    }

    // --- 1. Create Initial Population ---
    srand(time(NULL));
    NeuralNetwork** population = create_initial_population(params->population_size, params->num_layers, params->architecture);
    if (!population) {
        fprintf(stderr, "Failed to create initial population.\n");
        return NULL;
    }

    printf("Created initial population of %d networks.\n", params->population_size);
    printf("Starting evolution for %d generations...\n", params->num_generations);

    // --- 2. Run Evolutionary Loop ---
    for (int gen = 0; gen < params->num_generations; gen++) {
        NetworkFitness* population_with_fitness = malloc(params->population_size * sizeof(NetworkFitness));
        if (!population_with_fitness) {
             fprintf(stderr, "Failed to allocate memory for population fitness.\n");
             break; // Exit loop
        }

        double best_accuracy_in_gen = 0.0;
        for (int i = 0; i < params->population_size; i++) {
            population_with_fitness[i].network = population[i];
            population_with_fitness[i].fitness = calculate_fitness(population[i], train_dataset, params->fitness_samples);
            if (population_with_fitness[i].fitness > best_accuracy_in_gen) {
                best_accuracy_in_gen = population_with_fitness[i].fitness;
            }
        }
        printf("Generation %d/%d | Best Accuracy: %.2f%%\n", gen + 1, params->num_generations, best_accuracy_in_gen * 100.0);

        int num_fittest;
        NetworkFitness* fittest_networks_info = select_fittest(population_with_fitness, params->population_size, &num_fittest, params->selection_type, params->tournament_size);

        NeuralNetwork** new_population = reproduce(fittest_networks_info, num_fittest, params->population_size, params->mutation_rate, params->mutation_chance);

        // Free old population (but not the networks themselves, as they are pointed to by fittest_networks_info)
        free(population);
        free(population_with_fitness);
        free(fittest_networks_info);
        population = new_population;
    }

    // --- 3. Find Best Network from the final population ---
    NeuralNetwork* best_net = NULL;
    double best_overall_accuracy = 0.0;
    for (int i = 0; i < params->population_size; i++) {
        double accuracy = calculate_fitness(population[i], train_dataset, train_dataset->num_items); // Final evaluation on full dataset
        if (accuracy > best_overall_accuracy) {
            best_overall_accuracy = accuracy;
            // We need to clone the best network, because the population will be freed.
            if (best_net) free_neural_network(best_net);
            best_net = clone_network(population[i]);
        }
    }

    printf("Evolution finished. Best accuracy: %.2f%%\n", best_overall_accuracy * 100.0);

    // --- 4. Cleanup ---
    for (int i = 0; i < params->population_size; i++) {
        free_neural_network(population[i]);
    }
    free(population);

    return best_net; // Caller is responsible for freeing this network
}

int gann_predict(const NeuralNetwork* net, const double* input_data) {
    if (!net || !input_data) {
        return -1; // Invalid input
    }

    // Create a matrix for the input data
    Matrix* input_matrix = create_matrix(1, net->architecture[0]);
    if (!input_matrix) {
        return -1; // Memory allocation failed
    }
    memcpy(input_matrix->data[0], input_data, net->architecture[0] * sizeof(double));

    // Perform the forward pass
    Matrix* output_matrix = forward_pass(net, input_matrix);
    if (!output_matrix) {
        free_matrix(input_matrix);
        return -1; // Forward pass failed
    }

    // Get the result
    int prediction = get_predicted_class(output_matrix);

    // Cleanup
    free_matrix(input_matrix);
    free_matrix(output_matrix);

    return prediction;
}

double gann_evaluate(const NeuralNetwork* net, const Dataset* dataset) {
    if (!net || !dataset) {
        return 0.0;
    }

    int correct_predictions = 0;
    for (int i = 0; i < dataset->num_items; i++) {
        int prediction = gann_predict(net, dataset->images->data[i]);
        int true_class = get_true_class(dataset->labels->data[i]);

        if (prediction == true_class) {
            correct_predictions++;
        }
    }

    return (double)correct_predictions / dataset->num_items;
}
