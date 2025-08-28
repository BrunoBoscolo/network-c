#include "minunit.h"
#include "../include/neural_network.h"
#include <math.h>

extern const double TEST_EPSILON;

// Test for matrix creation
const char* test_matrix_creation() {
    Matrix* m = create_matrix(2, 3);
    mu_assert("Matrix creation failed to allocate", m != NULL);
    mu_assert("Incorrect number of rows", m->rows == 2);
    mu_assert("Incorrect number of columns", m->cols == 3);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            mu_assert("Matrix not initialized to zero", m->data[i][j] == 0.0);
        }
    }

    free_matrix(m);
    return NULL;
}

// Test for matrix dot product
const char* test_matrix_dot_product() {
    Matrix* m1 = create_matrix(2, 3);
    Matrix* m2 = create_matrix(3, 2);

    // Initialize m1: [[1, 2, 3], [4, 5, 6]]
    m1->data[0][0] = 1; m1->data[0][1] = 2; m1->data[0][2] = 3;
    m1->data[1][0] = 4; m1->data[1][1] = 5; m1->data[1][2] = 6;

    // Initialize m2: [[7, 8], [9, 10], [11, 12]]
    m2->data[0][0] = 7;  m2->data[0][1] = 8;
    m2->data[1][0] = 9;  m2->data[1][1] = 10;
    m2->data[2][0] = 11; m2->data[2][1] = 12;

    Matrix* result = dot_product(m1, m2);
    mu_assert("Dot product failed", result != NULL);
    mu_assert("Dot product result has wrong rows", result->rows == 2);
    mu_assert("Dot product result has wrong cols", result->cols == 2);

    // Expected result: [[58, 64], [139, 154]]
    mu_assert("Dot product calculation wrong at (0,0)", fabs(result->data[0][0] - 58) < TEST_EPSILON);
    mu_assert("Dot product calculation wrong at (0,1)", fabs(result->data[0][1] - 64) < TEST_EPSILON);
    mu_assert("Dot product calculation wrong at (1,0)", fabs(result->data[1][0] - 139) < TEST_EPSILON);
    mu_assert("Dot product calculation wrong at (1,1)", fabs(result->data[1][1] - 154) < TEST_EPSILON);

    free_matrix(m1);
    free_matrix(m2);
    free_matrix(result);
    return NULL;
}
