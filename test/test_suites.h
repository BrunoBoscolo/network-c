#ifndef TEST_SUITES_H
#define TEST_SUITES_H

// test_matrix.c
const char* test_matrix_creation();
const char* test_matrix_dot_product();
const char* test_matrix_errors();

// test_neural_network.c
const char* test_nn_creation();
const char* test_nn_forward_pass();
const char* test_gaussian_mutation();
const char* test_nn_errors();

// test_persistence.c
const char* test_save_and_load_network();
const char* test_persistence_errors();

// test_evolution.c
const char* test_crossover();
const char* test_single_point_crossover();
const char* test_two_point_crossover();

// test_backpropagation.c
const char* test_backprop_overfit_single_instance();

// test_optimizers.c
const char* optimizers_test_suite();

// test_genetic_operators.c
const char* genetic_operators_suite();

// Add declarations for other test suites here

// A function to run all test suites
const char* run_all_tests();

#endif // TEST_SUITES_H
