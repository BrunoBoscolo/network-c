#ifndef MATRIX_H
#define MATRIX_H

// Represents a 2D matrix
typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

// --- Matrix Operations ---

/**
 * @brief Creates a new matrix with all elements initialized to zero.
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @return A pointer to the newly created Matrix, or NULL on failure.
 */
Matrix* create_matrix(int rows, int cols);

/**
 * @brief Frees the memory allocated for a matrix.
 * @param m The matrix to free.
 */
void free_matrix(Matrix* m);

/**
 * @brief Prints the contents of a matrix to the console.
 * @param m The matrix to print.
 */
void print_matrix(const Matrix* m);

/**
 * @brief Computes the dot product of two matrices.
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @return A new matrix containing the result of the dot product, or NULL on failure.
 */
Matrix* dot_product(const Matrix* m1, const Matrix* m2);

/**
 * @brief Adds a bias vector to each row of a matrix.
 * @param m The matrix to modify.
 * @param bias The bias vector (must be a 1xN matrix).
 */
void add_bias(Matrix* m, const Matrix* bias);

#endif // MATRIX_H
