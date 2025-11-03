#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "gann_docs.h"

#include "minunit.h"
#include <stdlib.h>

const char* test_gann_get_doc() {
    char* doc_en = gann_get_doc("gann_seed_rng", "en");
    mu_assert("doc_en should not be NULL", doc_en != NULL);
    mu_assert("doc_en should contain the correct string", strstr(doc_en, "Seeds the random number generator") != NULL);
    free(doc_en);

    char* doc_br = gann_get_doc("gann_seed_rng", "br");
    mu_assert("doc_br should not be NULL", doc_br != NULL);
    mu_assert("doc_br should contain the correct string", strstr(doc_br, "Semeia o gerador de números aleatórios") != NULL);
    free(doc_br);

    char* doc_null = gann_get_doc("non_existent_function", "en");
    mu_assert("doc_null should be NULL", doc_null == NULL);
    free(doc_null);
    return NULL;
}

const char* test_gann_docs_suite() {
    mu_run_test(test_gann_get_doc);
    return NULL;
}
