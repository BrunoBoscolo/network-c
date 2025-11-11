#include "minunit.h"
#include "gann_docs.h"
#include "gann_errors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* test_get_doc_valid() {
    char* doc = gann_get_doc("gann_train", "en");
    mu_assert_string_eq("Trains a new neural network using a genetic algorithm. This function encapsulates the entire genetic algorithm training loop, including population initialization, evaluation, selection, crossover, and mutation.", doc);
    free(doc);
    return NULL;
}

char* test_get_doc_valid_br() {
    char* doc = gann_get_doc("gann_train", "br");
    mu_assert_string_eq("Treina uma nova rede neural usando um algoritmo genético. Esta função encapsula todo o ciclo de treinamento do algoritmo genético, incluindo a inicialização da população, avaliação, seleção, cruzamento e mutação.", doc);
    free(doc);
    return NULL;
}

char* test_get_doc_invalid_function() {
    char* doc = gann_get_doc("invalid_function", "en");
    mu_check(doc == NULL);
    mu_check(gann_get_last_error() == GANN_ERROR_DOC_NOT_FOUND);
    return NULL;
}

char* test_get_doc_invalid_lang() {
    char* doc = gann_get_doc("gann_train", "invalid_lang");
    mu_check(doc == NULL);
    mu_check(gann_get_last_error() == GANN_ERROR_FILE_READ);
    return NULL;
}

char* gann_docs_test_suite() {
    mu_run_test(test_get_doc_valid);
    mu_run_test(test_get_doc_valid_br);
    mu_run_test(test_get_doc_invalid_function);
    mu_run_test(test_get_doc_invalid_lang);
    return NULL;
}
