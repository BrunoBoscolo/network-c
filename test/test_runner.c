#include "minunit.h"
#include "test_suites.h"

int tests_run = 0;
const double TEST_EPSILON = 1e-9;

const char* all_suites() {
    // Run tests from test_matrix.c
    mu_run_test(test_matrix_creation);
    mu_run_test(test_matrix_dot_product);

    // Run tests from test_neural_network.c
    mu_run_test(test_nn_creation);
    mu_run_test(test_nn_forward_pass);

    // Run tests from test_persistence.c
    mu_run_test(test_save_and_load_network);

    // Run tests from test_evolution.c
    mu_run_test(test_crossover);

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
