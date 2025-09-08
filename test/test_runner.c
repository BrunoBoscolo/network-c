#include "minunit.h"
#include "test_suites.h"

int tests_run = 0;
const double TEST_EPSILON = 1e-9;

const char* all_suites() {
    // Run tests from test_matrix.c
    mu_run_test(test_matrix_creation);
    mu_run_test(test_matrix_dot_product);
    mu_run_test(test_matrix_errors);

    // Run tests from test_neural_network.c
    mu_run_test(test_nn_creation);
    mu_run_test(test_nn_forward_pass);
    mu_run_test(test_gaussian_mutation);
    mu_run_test(test_nn_errors);

    // Run tests from test_persistence.c
    mu_run_test(test_save_and_load_network);
    mu_run_test(test_persistence_errors);

    // Run tests from test_evolution.c
    mu_run_test(test_crossover);
    mu_run_test(test_single_point_crossover);
    mu_run_test(test_two_point_crossover);

    // Run tests from test_backpropagation.c
    mu_run_test(test_calculate_mse);
    mu_run_test(test_backprop_overfit_single_instance);
    mu_run_test(test_backprop_overfit_single_instance_adam);
    mu_run_test(test_backprop_overfit_single_instance_rmsprop);

    // Run tests from test_optimizers.c
    mu_run_test(optimizers_test_suite);

    // Run tests from test_genetic_operators.c
    mu_run_test(genetic_operators_suite);

    // Run tests from test_data_loader.c
    mu_run_test(data_loader_test_suite);

    // Run tests from test_gann_errors.c
    mu_run_test(gann_errors_test_suite);

    return NULL;
}

int main() {
    const char *result = all_suites();
    if (result != NULL) {
        printf("TEST FAILED: %s\n", result);
    } else {
        printf("ALL TESTS PASSED\n");
    }
    printf("Tests run: %d\n", tests_run);

    return result != NULL;
}
