#ifndef GANN_DOCS_H
#define GANN_DOCS_H

/**
 * @brief Retrieves the documentation for a given function.
 * @param function_name The name of the function to get documentation for.
 * @param lang The language to get the documentation in (e.g., "en", "br").
 * @return A dynamically allocated string containing the documentation for the function,
 *         or NULL if the function is not found. The caller is responsible for freeing this string.
 */
char* gann_get_doc(const char* function_name, const char* lang);

#endif // GANN_DOCS_H
