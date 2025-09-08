#include "minunit.h"
#include "test_suites.h"
#include "../include/data_loader.h"
#include <stdlib.h>

extern const double TEST_EPSILON;

const char* test_dummy_dataset_creation() {
    Dataset* ds = create_dummy_dataset(10);
    mu_assert("create_dummy_dataset should not return NULL", ds != NULL);
    mu_assert("Dataset should have 10 items", ds->num_items == 10);
    mu_assert("Images matrix should not be NULL", ds->images != NULL);
    mu_assert("Labels matrix should not be NULL", ds->labels != NULL);
    mu_assert("Images matrix should have 10 rows", ds->images->rows == 10);
    mu_assert("Labels matrix should have 10 rows", ds->labels->rows == 10);
    mu_assert("Images matrix should have correct number of columns", ds->images->cols == MNIST_IMAGE_SIZE);
    mu_assert("Labels matrix should have correct number of columns", ds->labels->cols == MNIST_NUM_CLASSES);
    free_dataset(ds);
    return NULL;
}

const char* test_load_mnist_valid() {
    Dataset* ds = load_mnist_dataset("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
    mu_assert("load_mnist_dataset should not return NULL for valid paths", ds != NULL);
    mu_assert("Dataset should have 60000 items", ds->num_items == 60000);
    free_dataset(ds);
    return NULL;
}

const char* test_load_mnist_invalid_path() {
    Dataset* ds = load_mnist_dataset("non/existent/path", "non/existent/path");
    mu_assert("load_mnist_dataset should return NULL for invalid paths", ds == NULL);
    return NULL;
}

const char* test_load_mnist_null_path() {
    Dataset* ds = load_mnist_dataset(NULL, NULL);
    mu_assert("load_mnist_dataset should return NULL for NULL paths", ds == NULL);
    return NULL;
}

const char* test_free_null_dataset() {
    free_dataset(NULL); // Should not crash
    return NULL;
}


const char* data_loader_test_suite() {
    mu_run_test(test_dummy_dataset_creation);
    mu_run_test(test_load_mnist_valid);
    mu_run_test(test_load_mnist_invalid_path);
    mu_run_test(test_load_mnist_null_path);
    mu_run_test(test_free_null_dataset);
    return NULL;
}
