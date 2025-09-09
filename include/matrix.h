#ifndef MATRIX_H
#define MATRIX_H

/**
 * @file matrix.h
 * @brief A basic 2D matrix library for neural network computations.
 * @details Provides functions for creating, manipulating, and performing
 * mathematical operations on 2D matrices of doubles.
 */

/**
 * @brief Represents a 2D matrix.
 */
typedef struct {
    int rows;      /**< The number of rows in the matrix. */
    int cols;      /**< The number of columns in the matrix. */
    double** data; /**< A 2D array holding the matrix elements. */
} Matrix;

// --- Matrix Operations ---

/**
 * @brief Creates a new matrix with all elements initialized to zero.
 * @details Allocates memory for a new `Matrix` struct and its underlying data array.
 * The caller is responsible for freeing the matrix using `free_matrix()`.
 * @param rows The number of rows in the new matrix.
 * @param cols The number of columns in the new matrix.
 * @return A pointer to the newly created `Matrix`, or `NULL` on failure.
 */
Matrix* create_matrix(int rows, int cols);

/**
 * @brief Frees the memory allocated for a matrix.
 * @details Deallocates the matrix's data array and the struct itself.
 * It is safe to pass `NULL` to this function.
 * @param m The matrix to free.
 */
void free_matrix(Matrix* m);

/**
 * @brief Prints the contents of a matrix to the console. Useful for debugging.
 * @param m The matrix to print.
 */
void print_matrix(const Matrix* m);

/**
 * @brief Computes the dot product of two matrices.
 * @details The number of columns in `m1` must equal the number of rows in `m2`.
 * @param m1 The first matrix (left operand).
 * @param m2 The second matrix (right operand).
 * @return A new matrix containing the result of the dot product. The caller is
 *         responsible for freeing this matrix. Returns `NULL` on failure.
 */
Matrix* dot_product(const Matrix* m1, const Matrix* m2);

/**
 * @brief Adds a bias vector (a row matrix) to each row of a matrix, in place.
 * @details The number of columns in `m` must equal the number of columns in `bias`.
 * `bias` must have exactly one row.
 * @param m The matrix to modify.
 * @param bias The bias vector (must be a 1xN matrix).
 */
void add_bias(Matrix* m, const Matrix* bias);

/**
 * @brief Creates a new matrix that is the transpose of the input matrix.
 * @param m The matrix to transpose.
 * @return A new matrix containing the transposed data. The caller is responsible
 *         for freeing this matrix. Returns `NULL` on failure.
 */
Matrix* matrix_transpose(const Matrix* m);

/**
 * @brief Performs element-wise multiplication (Hadamard product) of two matrices.
 * @details The matrices must have the same dimensions.
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @return A new matrix containing the result of the element-wise multiplication.
 *         The caller is responsible for freeing this matrix. Returns `NULL` on failure.
 */
Matrix* matrix_elementwise_multiply(const Matrix* m1, const Matrix* m2);

/**
 * @brief Subtracts the second matrix from the first, element by element.
 * @details The matrices must have the same dimensions.
 * @param m1 The matrix to subtract from (minuend).
 * @param m2 The matrix to subtract (subtrahend).
 * @return A new matrix containing the result of the subtraction. The caller is
 *         responsible for freeing this matrix. Returns `NULL` on failure.
 */
Matrix* matrix_subtract(const Matrix* m1, const Matrix* m2);

/**
 * @brief Adds two matrices, element by element.
 * @details The matrices must have the same dimensions.
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @return A new matrix containing the result of the addition. The caller is
 *         responsible for freeing this matrix. Returns `NULL` on failure.
 */
Matrix* matrix_add(const Matrix* m1, const Matrix* m2);

/**
 * @brief Scales a matrix by multiplying every element by a scalar value.
 * @param m The matrix to scale.
 * @param scalar The scalar value to multiply each element by.
 * @return A new matrix containing the scaled data. The caller is responsible
 *         for freeing this matrix. Returns `NULL` on failure.
 */
Matrix* matrix_scale(const Matrix* m, double scalar);

/**
 * @brief Creates a matrix from a flat, 1D array of data.
 * @param array The 1D array of data, assumed to be in row-major order.
 * @param rows The number of rows for the new matrix.
 * @param cols The number of columns for the new matrix.
 * @return A new matrix containing the data from the array. The caller is
 *         responsible for freeing this matrix. Returns `NULL` on failure.
 */
Matrix* matrix_from_array(const double* array, int rows, int cols);

/**
 * @brief Creates a deep copy of a matrix.
 * @param m The matrix to copy.
 * @return A new matrix that is an exact copy of the original. The caller is
 *         responsible for freeing this matrix. Returns `NULL` on failure.
 */
Matrix* matrix_copy(const Matrix* m);

/**
 * @brief Extracts a single row from a matrix and returns it as a new 1xN matrix.
 * @param m The matrix to extract the row from.
 * @param row The index of the row to extract (0-based).
 * @return A new matrix containing the data of the specified row. The caller is
 *         responsible for freeing this matrix. Returns `NULL` on failure.
 */
Matrix* matrix_get_row(const Matrix* m, int row);

/**
 * @brief Copies the data from a source matrix to a destination matrix.
 * @details This function only copies the `data` field. It assumes that the
 * destination matrix is already allocated and that both matrices have
 * identical dimensions.
 * @param dest The destination matrix.
 * @param src The source matrix.
 */
void matrix_copy_data(Matrix* dest, const Matrix* src);


#endif // MATRIX_H
