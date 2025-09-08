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

/**
 * @brief Creates a new matrix that is the transpose of the input matrix.
 * @param m The matrix to transpose.
 * @return A new matrix containing the transposed data.
 */
Matrix* matrix_transpose(const Matrix* m);

/**
 * @brief Performs element-wise multiplication (Hadamard product) of two matrices.
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @return A new matrix containing the result of the element-wise multiplication.
 */
Matrix* matrix_elementwise_multiply(const Matrix* m1, const Matrix* m2);

/**
 * @brief Subtracts the second matrix from the first matrix.
 * @param m1 The matrix to subtract from.
 * @param m2 The matrix to subtract.
 * @return A new matrix containing the result of the subtraction.
 */
Matrix* matrix_subtract(const Matrix* m1, const Matrix* m2);

/**
 * @brief Adds two matrices.
 * @param m1 The first matrix.
 * @param m2 The second matrix.
 * @return A new matrix containing the result of the addition.
 */
Matrix* matrix_add(const Matrix* m1, const Matrix* m2);

/**
 * @brief Scales a matrix by a scalar value.
 * @param m The matrix to scale.
 * @param scalar The scalar value to multiply each element by.
 * @return A new matrix containing the scaled data.
 */
Matrix* matrix_scale(const Matrix* m, double scalar);

/**
 * @brief Creates a matrix from a 1D array.
 * @param array The 1D array of data.
 * @param rows The number of rows for the new matrix.
 * @param cols The number of columns for the new matrix.
 * @return A new matrix containing the data from the array.
 */
Matrix* matrix_from_array(const double* array, int rows, int cols);

/**
 * @brief Creates a deep copy of a matrix.
 * @param m The matrix to copy.
 * @return A new matrix that is a copy of the original.
 */
Matrix* matrix_copy(const Matrix* m);

/**
 * @brief Extracts a single row from a matrix.
 * @param m The matrix to extract the row from.
 * @param row The index of the row to extract.
 * @return A new matrix containing the data of the specified row.
 */
Matrix* matrix_get_row(const Matrix* m, int row);

/**
 * @brief Copies the data from one matrix to another, assuming dimensions match.
 * @param dest The destination matrix.
 * @param src The source matrix.
 */
void matrix_copy_data(Matrix* dest, const Matrix* src);


#endif // MATRIX_H
