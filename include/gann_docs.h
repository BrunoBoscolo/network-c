#ifndef GANN_DOCS_H
#define GANN_DOCS_H

/**
 * @brief Retrieves the documentation for a specific function.
 * @details This function reads the documentation from a JSON file and returns
 * the documentation for the specified function.
 * @param function_name The name of the function to retrieve documentation for.
 * @param lang The language of the documentation to retrieve.
 * @return A string containing the documentation for the specified function.
 * The caller is responsible for freeing this string.
 * @return `NULL` on failure. If `NULL` is returned, call `gann_get_last_error()`
 * to get the specific error code.
 */
char* gann_get_doc(const char* function_name, const char* lang);

#endif // GANN_DOCS_H
