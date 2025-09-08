#include "gann_errors.h"
#include <stdio.h> // For NULL

// Define the thread-local variable to store the last error.
// It is initialized to GANN_SUCCESS by default.
static GANN_THREAD_LOCAL GannError g_last_error = GANN_SUCCESS;

// --- Public API Functions ---

GannError gann_get_last_error(void) {
    return g_last_error;
}

const char* gann_error_to_string(GannError error_code) {
    switch (error_code) {
        case GANN_SUCCESS:
            return "Success";
        case GANN_ERROR_UNKNOWN:
            return "An unknown error occurred";
        case GANN_ERROR_NULL_ARGUMENT:
            return "A required pointer argument was NULL";
        case GANN_ERROR_ALLOC_FAILED:
            return "Memory allocation failed";
        case GANN_ERROR_INVALID_PARAM:
            return "Invalid parameter provided to a function";
        case GANN_ERROR_FILE_OPEN:
            return "Failed to open file";
        case GANN_ERROR_FILE_READ:
            return "Failed to read from file";
        case GANN_ERROR_FILE_WRITE:
            return "Failed to write to file";
        case GANN_ERROR_INVALID_ARCHITECTURE:
            return "Invalid neural network architecture";
        case GANN_ERROR_INVALID_DIMENSIONS:
            return "Mismatched matrix or vector dimensions";
        case GANN_ERROR_INDEX_OUT_OF_BOUNDS:
            return "Index is out of bounds";
        case GANN_ERROR_INVALID_FILE_FORMAT:
            return "Invalid or corrupted file format";
        default:
            return "Unrecognized error code";
    }
}

// --- Internal Functions ---

void gann_set_error(GannError error_code) {
    g_last_error = error_code;
}
