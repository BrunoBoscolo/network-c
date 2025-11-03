#include <stdio.h>
#include "gann_docs.h"

int main() {
    const char* doc1 = gann_get_doc("gann_train");
    if (doc1) {
        printf("Documentation for gann_train:\n%s\n\n", doc1);
    } else {
        printf("Documentation for gann_train not found.\n\n");
    }

    const char* doc2 = gann_get_doc("nn_forward_pass");
    if (doc2) {
        printf("Documentation for nn_forward_pass:\n%s\n\n", doc2);
    } else {
        printf("Documentation for nn_forward_pass not found.\n\n");
    }

    const char* doc3 = gann_get_doc("non_existent_function");
    if (doc3) {
        printf("Documentation for non_existent_function:\n%s\n\n", doc3);
    } else {
        printf("Documentation for non_existent_function not found.\n\n");
    }

    return 0;
}
