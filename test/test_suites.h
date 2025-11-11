#ifndef TEST_SUITES_H
#define TEST_SUITES_H

#include "minunit.h"

// --- Test Suites ---
const char* test_matrix_creation();
const char* test_matrix_dot_product();
const char* test_matrix_errors();
const char* test_nn_creation();
const char* test_nn_forward_pass();
const char* test_gaussian_mutation();
const char* test_nn_errors();
const char* test_save_and_load_network();
const char* test_persistence_errors();
const char* test_crossover();
const char* test_single_point_crossover();
const char* test_two_point_crossover();
const char* test_calculate_mse();
const char* test_backprop_overfit_single_instance();
const char* test_backprop_overfit_single_instance_adam();
const char* test_backprop_overfit_single_instance_rmsprop();
const char* test_backprop_early_stopping();
const char* optimizers_test_suite();
const char* genetic_operators_suite();
const char* data_loader_test_suite();
const char* gann_errors_test_suite();
const char* gann_docs_test_suite();
char* test_get_doc_valid();
char* test_get_doc_valid_br();
char* test_get_doc_invalid_function();
char* test_get_doc_invalid_lang();

#endif // TEST_SUITES_H
