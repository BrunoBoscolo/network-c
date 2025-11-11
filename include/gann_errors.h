#ifndef GANN_ERRORS_H
#define GANN_ERRORS_H

/**
 * @file gann_errors.h
 * @brief Thread-safe error handling for the GANN library.
 * @details Provides an enumeration of possible error codes and functions to
 * retrieve and interpret them. Errors are stored in a thread-local variable,
 * making the error handling mechanism safe for multi-threaded applications.
 */

#if __STDC_VERSION__ >= 201112L
#define GANN_THREAD_LOCAL _Thread_local
#elif defined(__GNUC__) || defined(__clang__)
#define GANN_THREAD_LOCAL __thread
#else
// Fallback for older compilers - not thread-safe
#define GANN_THREAD_LOCAL
#endif


// --- Error Code Enum ---

/**
 * @brief Represents all possible error codes in the GANN library.
 */
typedef enum {
    GANN_SUCCESS = 0,               /**< The operation completed successfully. */
    GANN_ERROR_UNKNOWN,             /**< An unknown or unspecified error occurred. */
    GANN_ERROR_NULL_ARGUMENT,       /**< A required pointer argument passed to a function was NULL. */
    GANN_ERROR_ALLOC_FAILED,        /**< A memory allocation (e.g., malloc, calloc) failed, likely due to insufficient memory. */
    GANN_ERROR_INVALID_PARAM,       /**< A function was called with an invalid parameter value (e.g., a negative size). */
    GANN_ERROR_FILE_OPEN,           /**< A file operation failed because the file could not be opened. */
    GANN_ERROR_FILE_READ,           /**< An error occurred while trying to read from a file. */
    GANN_ERROR_FILE_WRITE,          /**< An error occurred while trying to write to a file. */
    GANN_ERROR_INVALID_ARCHITECTURE,/**< The specified neural network architecture is invalid (e.g., contains fewer than 2 layers). */
    GANN_ERROR_INVALID_DIMENSIONS,  /**< An operation could not be completed due to mismatched matrix or vector dimensions. */
    GANN_ERROR_INDEX_OUT_OF_BOUNDS, /**< An index used to access an array or matrix was outside the valid range. */
    GANN_ERROR_INVALID_FILE_FORMAT, /**< A file being loaded has an invalid or corrupted format. */
    GANN_ERROR_DOC_NOT_FOUND        /**< The requested documentation was not found. */
} GannError;


// --- Public Error Handling Functions ---

/**
 * @brief Gets the last error that occurred on the calling thread.
 * @details When a library function fails (e.g., returns `NULL` or a status code
 * indicating an error), this function can be called to retrieve the specific
 * error code, which provides more details about the cause of the failure.
 * @return The `GannError` code for the last error that occurred on the current thread.
 */
GannError gann_get_last_error(void);

/**
 * @brief Converts a `GannError` code into a human-readable, null-terminated string.
 * @param error_code The error code to convert.
 * @return A constant string describing the error, suitable for logging or displaying to a user.
 */
const char* gann_error_to_string(GannError error_code);


// --- Internal Error Handling Functions (Do not use directly) ---

/**
 * @internal
 * @brief Sets the last error code for the calling thread.
 * @details This function is used internally by the library to report errors.
 * It should not be called by user code.
 * @param error_code The error code to set as the last error for the current thread.
 */
void gann_set_error(GannError error_code);


#endif // GANN_ERRORS_H
