#ifndef GANN_ERRORS_H
#define GANN_ERRORS_H

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
    GANN_SUCCESS = 0,               /**< The operation was successful. */
    GANN_ERROR_UNKNOWN,             /**< An unknown or unspecified error occurred. */
    GANN_ERROR_NULL_ARGUMENT,       /**< A required pointer argument was NULL. */
    GANN_ERROR_ALLOC_FAILED,        /**< A memory allocation (e.g., malloc, calloc) failed. */
    GANN_ERROR_INVALID_PARAM,       /**< A function was called with an invalid parameter value (e.g., negative size). */
    GANN_ERROR_FILE_OPEN,           /**< Failed to open a file. */
    GANN_ERROR_FILE_READ,           /**< Failed to read from a file. */
    GANN_ERROR_FILE_WRITE,          /**< Failed to write to a file. */
    GANN_ERROR_INVALID_ARCHITECTURE,/**< The neural network architecture is invalid (e.g., < 2 layers). */
    GANN_ERROR_INVALID_DIMENSIONS,  /**< Mismatched matrix or vector dimensions for an operation. */
    GANN_ERROR_INDEX_OUT_OF_BOUNDS, /**< An index was outside the valid range. */
    GANN_ERROR_INVALID_FILE_FORMAT  /**< The format of a file being loaded is invalid or corrupted. */
} GannError;


// --- Public Error Handling Functions ---

/**
 * @brief Gets the last error that occurred on the calling thread.
 *
 * When a library function fails (e.g., returns NULL), this function can be
 * called to retrieve the specific error code that provides more details
 * about the failure.
 *
 * @return The last error code for the current thread.
 */
GannError gann_get_last_error(void);

/**
 * @brief Converts a GannError code into a human-readable, null-terminated string.
 *
 * @param error_code The error code to convert.
 * @return A constant string describing the error.
 */
const char* gann_error_to_string(GannError error_code);


// --- Internal Error Handling Functions (Do not use directly) ---

/**
 * @internal
 * @brief Sets the last error code for the calling thread.
 *
 * This function is used internally by the library to report errors.
 * It is not intended to be called by the user.
 *
 * @param error_code The error code to set.
 */
void gann_set_error(GannError error_code);


#endif // GANN_ERRORS_H
