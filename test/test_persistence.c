#include "minunit.h"
#include "neural_network.h"
#include "gann_errors.h"
#include <stdio.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

extern const double TEST_EPSILON;

const char* test_save_and_load_network() {
    int architecture[] = {2, 3, 1};
    NeuralNetwork* original_net = nn_create(3, architecture, SIGMOID, SIGMOID);

    // Set some specific values to test
    original_net->weights[0]->data[0][0] = 0.123;
    original_net->biases[0]->data[0][0] = 0.456;

    const char* filepath = "test_network.dat";
    int result = nn_save(original_net, filepath);
    mu_assert("Failed to save network", result == 1);
    mu_assert("nn_save should set GANN_SUCCESS", gann_get_last_error() == GANN_SUCCESS);

    NeuralNetwork* loaded_net = nn_load(filepath);
    mu_assert("Failed to load network", loaded_net != NULL);
    mu_assert("nn_load should set GANN_SUCCESS", gann_get_last_error() == GANN_SUCCESS);


    // Compare architecture
    mu_assert("Loaded network has wrong number of layers", original_net->num_layers == loaded_net->num_layers);
    for (int i = 0; i < original_net->num_layers; i++) {
        mu_assert("Loaded network has wrong architecture", original_net->architecture[i] == loaded_net->architecture[i]);
    }

    // Compare weights and biases
    for (int i = 0; i < original_net->num_layers - 1; i++) {
        for (int r = 0; r < original_net->weights[i]->rows; r++) {
            for (int c = 0; c < original_net->weights[i]->cols; c++) {
                double diff = fabs(original_net->weights[i]->data[r][c] - loaded_net->weights[i]->data[r][c]);
                mu_assert("Loaded network has wrong weights", diff < TEST_EPSILON);
            }
        }
        for (int c = 0; c < original_net->biases[i]->cols; c++) {
            double diff = fabs(original_net->biases[i]->data[0][c] - loaded_net->biases[i]->data[0][c]);
            mu_assert("Loaded network has wrong biases", diff < TEST_EPSILON);
        }
    }

    nn_free(original_net);
    nn_free(loaded_net);
    remove(filepath);

    return NULL;
}

// Test for persistence error handling
const char* test_persistence_errors() {
    // --- Suppress stderr for this test ---
    int stderr_copy = dup(STDERR_FILENO);
    int dev_null = open("/dev/null", O_WRONLY);
    dup2(dev_null, STDERR_FILENO);
    close(dev_null);

    // Test nn_load with a non-existent file
    NeuralNetwork* net = nn_load("non_existent_file.dat");
    mu_assert("nn_load should fail for non-existent file", net == NULL);
    mu_assert("nn_load should set GANN_ERROR_FILE_OPEN", gann_get_last_error() == GANN_ERROR_FILE_OPEN);

    // Test nn_save with a NULL network
    int result = nn_save(NULL, "test_save_null.dat");
    mu_assert("nn_save should fail for NULL network", result == 0);
    mu_assert("nn_save should set GANN_ERROR_NULL_ARGUMENT", gann_get_last_error() == GANN_ERROR_NULL_ARGUMENT);

    // Test nn_save to an invalid path
    int arch[] = {1, 1};
    net = nn_create(2, arch, SIGMOID, SIGMOID);
    // This will fail on most systems as you can't create a file with the name of a directory that exists.
    result = nn_save(net, ".");
    mu_assert("nn_save should fail for invalid path", result == 0);
    mu_assert("nn_save should set GANN_ERROR_FILE_OPEN", gann_get_last_error() == GANN_ERROR_FILE_OPEN);
    nn_free(net);

    // Test loading from a corrupted/invalid file
    FILE* f = fopen("corrupted.dat", "w");
    if (f) {
        fprintf(f, "this is not a valid network file");
        fclose(f);
        net = nn_load("corrupted.dat");
        mu_assert("nn_load should fail for corrupted file", net == NULL);
        GannError err = gann_get_last_error();
        mu_assert("nn_load should set an error for corrupted file", err == GANN_ERROR_FILE_READ || err == GANN_ERROR_INVALID_FILE_FORMAT);
        remove("corrupted.dat");
    }

    // --- Restore stderr ---
    dup2(stderr_copy, STDERR_FILENO);
    close(stderr_copy);

    return NULL;
}
