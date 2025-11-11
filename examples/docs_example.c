#include "gann.h"
#include "gann_docs.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    const char* functions[] = {
        "gann_seed_rng",
        "gann_create_default_params",
        "gann_evolve",
        "gann_train",
        "gann_train_with_backprop",
        "gann_predict",
        "gann_evaluate",
        "nn_create",
        "nn_init",
        "nn_free",
        "nn_forward_pass",
        "nn_clone",
        "nn_save",
        "nn_load",
        "backpropagate",
        "update_weights_sgd",
        "update_weights_rmsprop",
        "update_weights_adam",
        "calculate_mse",
        "crossover",
        "load_mnist_dataset",
        "create_dummy_dataset",
        "create_dummy_dataset_with_label",
        "split_dataset",
        "free_dataset",
        "evo_create_initial_population",
        "evo_reproduce",
        "gann_get_last_error",
        "gann_error_to_string",
        "create_matrix",
        "free_matrix",
        "print_matrix",
        "dot_product",
        "add_bias",
        "matrix_transpose",
        "matrix_elementwise_multiply",
        "matrix_subtract",
        "matrix_add",
        "matrix_scale",
        "matrix_from_array",
        "matrix_copy",
        "matrix_get_row",
        "matrix_copy_data",
        "mutate_network",
        "select_fittest",
        NULL
    };

    for (int i = 0; functions[i] != NULL; i++) {
        char* doc = gann_get_doc(functions[i], "en");
        if (doc) {
            printf("--- Documentation for %s ---\n", functions[i]);
            printf("%s\n\n", doc);
            free(doc);
        } else {
            printf("--- No documentation found for %s ---\n\n", functions[i]);
        }
    }

    return 0;
}
