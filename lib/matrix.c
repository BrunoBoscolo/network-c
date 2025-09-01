#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

// --- Matrix Operations Implementation ---

// Creates and allocates memory for a new matrix
Matrix* create_matrix(int rows, int cols) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) return NULL;

    m->rows = rows;
    m->cols = cols;
    m->data = (double**)malloc(rows * sizeof(double*));
    if (!m->data) {
        free(m);
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        m->data[i] = (double*)calloc(cols, sizeof(double));
        if (!m->data[i]) {
            // Rollback allocation on failure
            for (int j = 0; j < i; j++) free(m->data[j]);
            free(m->data);
            free(m);
            return NULL;
        }
    }
    return m;
}

// Frees the memory of a matrix
void free_matrix(Matrix* m) {
    if (!m) return;
    for (int i = 0; i < m->rows; i++) {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

// Prints the matrix data (for debugging)
void print_matrix(const Matrix* m) {
    if (!m) return;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%f ", m->data[i][j]);
        }
        printf("\n");
    }
}

// Computes the dot product of two matrices
Matrix* dot_product(const Matrix* m1, const Matrix* m2) {
    if (m1->cols != m2->rows) return NULL;

    Matrix* result = create_matrix(m1->rows, m2->cols);
    if (!result) return NULL;

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m2->cols; j++) {
            for (int k = 0; k < m1->cols; k++) {
                result->data[i][j] += m1->data[i][k] * m2->data[k][j];
            }
        }
    }
    return result;
}

// Adds a bias vector to each row of a matrix
void add_bias(Matrix* m, const Matrix* bias) {
    if (m->cols != bias->cols || bias->rows != 1) return;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] += bias->data[0][j];
        }
    }
}
