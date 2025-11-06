#include <stdio.h>
#include <stdlib.h> // Required for free()
#include "gann_docs.h"

int main() {
    char* doc1 = gann_get_doc("gann_train", "pt");
    if (doc1) {
        printf("Documentation for gann_train:\n%s\n\n", doc1);
        free(doc1); // Free the dynamically allocated memory
    } else {
        printf("Documentation for gann_train not found.\n\n");
    }

    char* doc2 = gann_get_doc("nn_forward_pass", "pt");
    if (doc2) {
        printf("Documentation for nn_forward_pass:\n%s\n\n", doc2);
        free(doc2); // Free the dynamically allocated memory
    } else {
        printf("Documentation for nn_forward_pass not found.\n\n");
    }

    char* doc3 = gann_get_doc("non_existent_function", "pt");
    if (doc3) {
        printf("Documentation for non_existent_function:\n%s\n\n", doc3);
        free(doc3); // Free the dynamically allocated memory
    } else {
        printf("Documentation for non_existent_function not found.\n\n");
    }

    return 0;
}
