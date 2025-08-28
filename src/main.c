#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "data_loader.h"
#include "evolution.h"
#include "neural_network.h"

// --- Helper: Find index of max value ---
int get_predicted_class(const Matrix *output) {
  int max_index = 0;
  for (int i = 1; i < output->cols; i++) {
    if (output->data[0][i] > output->data[0][max_index]) {
      max_index = i;
    }
  }
  return max_index;
}

// --- Fitness Function (Accuracy) ---
// Note: Evaluating on the full dataset is slow. We use a subset.
double calculate_fitness(NeuralNetwork *network, const Dataset *dataset,
                         int num_samples) {
  int correct_predictions = 0;
  if (num_samples > dataset->num_items) {
    num_samples = dataset->num_items;
  }

  for (int i = 0; i < num_samples; i++) {
    // Create a temporary matrix for a single image input
    Matrix *input = create_matrix(1, MNIST_IMAGE_SIZE);
    if (!input)
      continue;
    memcpy(input->data[0], dataset->images->data[i],
           MNIST_IMAGE_SIZE * sizeof(double));

    Matrix *output = forward_pass(network, input);
    if (!output) {
      free_matrix(input);
      continue;
    }

    int predicted_class = get_predicted_class(output);

    int true_class = 0;
    for (int j = 0; j < MNIST_NUM_CLASSES; j++) {
      if (dataset->labels->data[i][j] == 1.0) {
        true_class = j;
        break;
      }
    }

    if (predicted_class == true_class) {
      correct_predictions++;
    }

    free_matrix(input);
    free_matrix(output);
  }

  return (double)correct_predictions / num_samples;
}

int main() {
  printf(
      "--- Starting MNIST Training with Genetic Algorithm (C Version) ---\n");

  // --- 1. Define Parameters ---
  const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 128, MNIST_NUM_CLASSES};
  const int NUM_LAYERS = sizeof(ARCHITECTURE) / sizeof(int);
#define POPULATION_SIZE 50
#define NUM_GENERATIONS 100
  const float MUTATION_RATE = 0.5f;
  const float MUTATION_CHANCE = 0.25f;

  // --- 2. Load MNIST Data ---
  Dataset *train_dataset = load_mnist_dataset("data/train-images.idx3-ubyte",
                                              "data/train-labels.idx1-ubyte");
  if (!train_dataset) {
    fprintf(stderr, "Failed to load training data.\n");
    return 1;
  }
  // The training set will be used for both training and fitness evaluation.

#define FITNESS_SAMPLES 1000 // Use 1000 samples for fitness eval

  // --- 3. Create Initial Population ---
  srand(time(NULL));
  NeuralNetwork **population =
      create_initial_population(POPULATION_SIZE, NUM_LAYERS, ARCHITECTURE);
  printf("Created initial population of %d networks.\n", POPULATION_SIZE);
  printf("Network architecture: [");
  for (int i = 0; i < NUM_LAYERS; i++)
    printf("%d%s", ARCHITECTURE[i], i == NUM_LAYERS - 1 ? "" : ", ");
  printf("]\n");
  printf("Using %d samples for fitness evaluation.\n", FITNESS_SAMPLES);
  printf("--------------------\n");

  // --- 4. Run Evolutionary Loop ---
  for (int gen = 0; gen < NUM_GENERATIONS; gen++) {
    NetworkFitness population_with_fitness[POPULATION_SIZE];
    double best_accuracy_in_gen = 0.0;

    for (int i = 0; i < POPULATION_SIZE; i++) {
      population_with_fitness[i].network = population[i];
      population_with_fitness[i].fitness =
          calculate_fitness(population[i], train_dataset, FITNESS_SAMPLES);
      if (population_with_fitness[i].fitness > best_accuracy_in_gen) {
        best_accuracy_in_gen = population_with_fitness[i].fitness;
      }
    }
    printf("Generation %d/%d | Best Accuracy: %.2f%%\n", gen + 1,
           NUM_GENERATIONS, best_accuracy_in_gen * 100.0);

    const SelectionType SELECTION_TYPE = TOURNAMENT;
    const int TOURNAMENT_SIZE = 4;
    int num_fittest;
    NetworkFitness *fittest_networks_info =
        select_fittest(population_with_fitness, POPULATION_SIZE, &num_fittest, SELECTION_TYPE, TOURNAMENT_SIZE);

    NeuralNetwork **new_population =
        reproduce(fittest_networks_info, num_fittest, POPULATION_SIZE,
                  MUTATION_RATE, MUTATION_CHANCE);

    for (int i = 0; i < POPULATION_SIZE; i++) {
      free_neural_network(population[i]);
    }
    free(population);
    free(fittest_networks_info);
    population = new_population;
  }

  printf("--------------------\n");
  // --- 5. Find Best Network and Save ---
  NeuralNetwork *best_net = NULL;
  double best_overall_accuracy = 0.0;
  for (int i = 0; i < POPULATION_SIZE; i++) {
    double accuracy =
        calculate_fitness(population[i], train_dataset, FITNESS_SAMPLES);
    if (accuracy > best_overall_accuracy) {
      best_overall_accuracy = accuracy;
      best_net = population[i];
    }
  }

  printf("Evolution finished.\n");
  printf("Best accuracy achieved after %d generations: %.2f%%\n",
         NUM_GENERATIONS, best_overall_accuracy * 100.0);

  if (best_net) {
    if (save_network(best_net, "trained_network.dat")) {
      printf("Best network saved to trained_network.dat\n");
    } else {
      fprintf(stderr, "Failed to save the best network.\n");
    }
  }

  // --- 6. Cleanup ---
  free_dataset(train_dataset);
  for (int i = 0; i < POPULATION_SIZE; i++) {
    free_neural_network(population[i]);
  }
  free(population);

  return 0;
}
