#include "minunit.h"
#include "selection.h"
#include "crossover.h"
#include "mutation.h"
#include "neural_network.h"
#include "evolution.h"

#define NUM_TEST_NETWORKS 10

static NeuralNetwork* test_networks[NUM_TEST_NETWORKS];
static NetworkFitness test_population[NUM_TEST_NETWORKS];

void test_setup_genetic_operators() {
    int architecture[] = {2, 3, 1};
    for (int i = 0; i < NUM_TEST_NETWORKS; i++) {
        test_networks[i] = create_neural_network(3, architecture, SIGMOID, SIGMOID);
        initialize_network(test_networks[i]);
        test_population[i].network = test_networks[i];
        test_population[i].fitness = (double)rand() / RAND_MAX;
    }
}

void test_teardown_genetic_operators() {
    for (int i = 0; i < NUM_TEST_NETWORKS; i++) {
        free_neural_network(test_networks[i]);
    }
}

const char* test_roulette_wheel_selection() {
    int num_fittest;
    NetworkFitness* fittest = select_fittest(test_population, NUM_TEST_NETWORKS, &num_fittest, ROULETTE_WHEEL_SELECTION, 0);
    mu_assert("Roulette wheel selection returned NULL", fittest != NULL);
    mu_assert("Roulette wheel selection did not select the correct number of individuals", num_fittest == NUM_TEST_NETWORKS / 2);
    free(fittest);
    return NULL;
}

const char* test_rank_selection() {
    int num_fittest;
    NetworkFitness* fittest = select_fittest(test_population, NUM_TEST_NETWORKS, &num_fittest, RANK_SELECTION, 0);
    mu_assert("Rank selection returned NULL", fittest != NULL);
    mu_assert("Rank selection did not select the correct number of individuals", num_fittest == NUM_TEST_NETWORKS / 2);
    free(fittest);
    return NULL;
}

const char* test_arithmetic_crossover() {
    NeuralNetwork* child = crossover(test_networks[0], test_networks[1], ARITHMETIC_CROSSOVER);
    mu_assert("Arithmetic crossover returned NULL", child != NULL);
    free_neural_network(child);
    return NULL;
}

const char* test_non_uniform_mutation() {
    NeuralNetwork* net = clone_network(test_networks[0]);
    mutate_network(net, 0.1, 0.1, NON_UNIFORM_MUTATION, 0.1, 0, 100, 0.1);
    // No assertion, just checking it runs without crashing
    free_neural_network(net);
    return NULL;
}

const char* test_adaptive_mutation() {
    NeuralNetwork* net = clone_network(test_networks[0]);
    mutate_network(net, 0.1, 0.1, ADAPTIVE_MUTATION, 0.1, 0, 100, 0.1);
    // No assertion, just checking it runs without crashing
    free_neural_network(net);
    return NULL;
}

const char* genetic_operators_suite() {
    test_setup_genetic_operators();
    mu_run_test(test_roulette_wheel_selection);
    mu_run_test(test_rank_selection);
    mu_run_test(test_arithmetic_crossover);
    mu_run_test(test_non_uniform_mutation);
    mu_run_test(test_adaptive_mutation);
    test_teardown_genetic_operators();
    return NULL;
}
