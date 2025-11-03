#ifndef GANN_DOCS_H
#define GANN_DOCS_H

/**
 * @brief Retrieves the documentation for a given function.
 * @param function_name The name of the function to get documentation for.
 * @return A string containing the documentation for the function, or NULL if the function is not found.
 */
const char* gann_get_doc(const char* function_name);

#endif // GANN_DOCS_H
