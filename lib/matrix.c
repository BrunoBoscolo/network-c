#include "matrix.h"
#include "gann_errors.h"
#include <stdio.h>
#include <stdlib.h>

// --- Matrix Operations Implementation ---

// Creates and allocates memory for a new matrix
Matrix* create_matrix(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        gann_set_error(GANN_ERROR_INVALID_PARAM);
        return NULL;
    }

    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
    if (!m) {
        gann_set_error(GANN_ERROR_ALLOC_FAILED);
        return NULL;
    }

    m->rows = rows;
    m->cols = cols;
    m->data = (double**)malloc(rows * sizeof(double*));
    if (!m->data) {
        free(m);
        gann_set_error(GANN_ERROR_ALLOC_FAILED);
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        m->data[i] = (double*)calloc(cols, sizeof(double));
        if (!m->data[i]) {
            // Rollback allocation on failure
            for (int j = 0; j < i; j++) free(m->data[j]);
            free(m->data);
            free(m);
            gann_set_error(GANN_ERROR_ALLOC_FAILED);
            return NULL;
        }
    }
    gann_set_error(GANN_SUCCESS);
    return m;
}

// Frees the memory of a matrix
void free_matrix(Matrix* m) {
    if (!m) return;
    if (m->data) {
        for (int i = 0; i < m->rows; i++) {
            free(m->data[i]);
        }
        free(m->data);
    }
    free(m);
}

// Prints the matrix data (for debugging)
void print_matrix(const Matrix* m) {
    if (!m) {
        printf("(Null Matrix)\n");
        return;
    }
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%f ", m->data[i][j]);
        }
        printf("\n");
    }
}

// Computes the dot product of two matrices
Matrix* dot_product(const Matrix* m1, const Matrix* m2) {
    if (!m1 || !m2) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    if (m1->cols != m2->rows) {
        gann_set_error(GANN_ERROR_INVALID_DIMENSIONS);
        return NULL;
    }

    Matrix* result = create_matrix(m1->rows, m2->cols);
    if (!result) return NULL; // create_matrix sets the error

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
    if (!m || !bias) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return;
    }
    if (m->cols != bias->cols || bias->rows != 1) {
        gann_set_error(GANN_ERROR_INVALID_DIMENSIONS);
        return;
    }
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] += bias->data[0][j];
        }
    }
    gann_set_error(GANN_SUCCESS);
}

// Creates a new matrix that is the transpose of the input matrix
Matrix* matrix_transpose(const Matrix* m) {
    if (!m) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    Matrix* result = create_matrix(m->cols, m->rows);
    if (!result) return NULL; // create_matrix sets the error

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[j][i] = m->data[i][j];
        }
    }
    return result;
}

// Performs element-wise multiplication (Hadamard product) of two matrices
Matrix* matrix_elementwise_multiply(const Matrix* m1, const Matrix* m2) {
    if (!m1 || !m2) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        gann_set_error(GANN_ERROR_INVALID_DIMENSIONS);
        return NULL;
    }

    Matrix* result = create_matrix(m1->rows, m1->cols);
    if (!result) return NULL; // create_matrix sets the error

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->data[i][j] = m1->data[i][j] * m2->data[i][j];
        }
    }
    return result;
}

// Subtracts the second matrix from the first matrix
Matrix* matrix_subtract(const Matrix* m1, const Matrix* m2) {
    if (!m1 || !m2) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        gann_set_error(GANN_ERROR_INVALID_DIMENSIONS);
        return NULL;
    }

    Matrix* result = create_matrix(m1->rows, m1->cols);
    if (!result) return NULL; // create_matrix sets the error

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->data[i][j] = m1->data[i][j] - m2->data[i][j];
        }
    }
    return result;
}

// Adds two matrices
Matrix* matrix_add(const Matrix* m1, const Matrix* m2) {
    if (!m1 || !m2) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        gann_set_error(GANN_ERROR_INVALID_DIMENSIONS);
        return NULL;
    }

    Matrix* result = create_matrix(m1->rows, m1->cols);
    if (!result) return NULL; // create_matrix sets the error

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            result->data[i][j] = m1->data[i][j] + m2->data[i][j];
        }
    }
    return result;
}

// Scales a matrix by a scalar value
Matrix* matrix_scale(const Matrix* m, double scalar) {
    if (!m) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    Matrix* result = create_matrix(m->rows, m->cols);
    if (!result) return NULL; // create_matrix sets the error

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[i][j] = m->data[i][j] * scalar;
        }
    }
    return result;
}

// Creates a matrix from a 1D array
Matrix* matrix_from_array(const double* array, int rows, int cols) {
    if (!array) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    Matrix* m = create_matrix(rows, cols);
    if (!m) return NULL; // create_matrix sets the error

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m->data[i][j] = array[i * cols + j];
        }
    }
    return m;
}

// Creates a deep copy of a matrix
Matrix* matrix_copy(const Matrix* m) {
    if (!m) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    Matrix* copy = create_matrix(m->rows, m->cols);
    if (!copy) return NULL; // create_matrix sets the error

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            copy->data[i][j] = m->data[i][j];
        }
    }
    return copy;
}

// Extracts a single row from a matrix
Matrix* matrix_get_row(const Matrix* m, int row) {
    if (!m) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    if (row < 0 || row >= m->rows) {
        gann_set_error(GANN_ERROR_INDEX_OUT_OF_BOUNDS);
        return NULL;
    }
    Matrix* result = create_matrix(1, m->cols);
    if (!result) return NULL; // create_matrix sets the error

    for (int j = 0; j < m->cols; j++) {
        result->data[0][j] = m->data[row][j];
    }
    return result;
}
