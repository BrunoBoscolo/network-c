#include "minunit.h"
#include "test_suites.h"
#include "../include/gann_errors.h"
#include <string.h>

const char* test_error_state_management() {
    // 1. Initial state should be SUCCESS
    mu_assert("Initial error state should be GANN_SUCCESS", gann_get_last_error() == GANN_SUCCESS);

    // 2. Set and get an error
    gann_set_error(GANN_ERROR_ALLOC_FAILED);
    mu_assert("gann_get_last_error should return the error set by gann_set_error", gann_get_last_error() == GANN_ERROR_ALLOC_FAILED);

    // 3. Set another error
    gann_set_error(GANN_ERROR_INVALID_DIMENSIONS);
    mu_assert("gann_get_last_error should reflect the latest error", gann_get_last_error() == GANN_ERROR_INVALID_DIMENSIONS);

    // 4. Reset to success
    gann_set_error(GANN_SUCCESS);
    mu_assert("Error state should be resettable to GANN_SUCCESS", gann_get_last_error() == GANN_SUCCESS);

    return NULL;
}

const char* test_error_to_string_conversion() {
    // Test a few specific, common errors
    mu_assert("String for GANN_SUCCESS is incorrect", strcmp(gann_error_to_string(GANN_SUCCESS), "Success") == 0);
    mu_assert("String for GANN_ERROR_NULL_ARGUMENT is incorrect", strcmp(gann_error_to_string(GANN_ERROR_NULL_ARGUMENT), "A required pointer argument was NULL") == 0);
    mu_assert("String for GANN_ERROR_FILE_OPEN is incorrect", strcmp(gann_error_to_string(GANN_ERROR_FILE_OPEN), "Failed to open file") == 0);

    // Test the last valid error code
    mu_assert("String for GANN_ERROR_INVALID_FILE_FORMAT is incorrect", strcmp(gann_error_to_string(GANN_ERROR_INVALID_FILE_FORMAT), "Invalid or corrupted file format") == 0);

    // Test an invalid error code
    GannError invalid_error = (GannError)999;
    mu_assert("String for an unrecognized error code is incorrect", strcmp(gann_error_to_string(invalid_error), "Unrecognized error code") == 0);

    return NULL;
}

const char* gann_errors_test_suite() {
    mu_run_test(test_error_state_management);
    mu_run_test(test_error_to_string_conversion);
    return NULL;
}
