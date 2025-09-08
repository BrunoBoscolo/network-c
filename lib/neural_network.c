#include "neural_network.h"
#include "matrix.h"
#include "gann_errors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// --- Private Activation Functions ---
static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double relu(double x) { return x > 0 ? x : 0; }
static double leaky_relu(double x) { return x > 0 ? x : 0.01 * x; }
static double sigmoid_derivative(double x) { double s = sigmoid(x); return s * (1 - s); }
static double relu_derivative(double x) { return x > 0 ? 1 : 0; }
static double leaky_relu_derivative(double x) { return x > 0 ? 1 : 0.01; }

// --- Public API Functions ---

void nn_apply_activation(Matrix* m, ActivationType activation_type) {
    if (!m) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return;
    }
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double* val = &m->data[i][j];
            switch (activation_type) {
                case SIGMOID: *val = sigmoid(*val); break;
                case RELU: *val = relu(*val); break;
                case LEAKY_RELU: *val = leaky_relu(*val); break;
            }
        }
    }
    gann_set_error(GANN_SUCCESS);
}

void nn_apply_activation_derivative(Matrix* m, ActivationType activation_type) {
    if (!m) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return;
    }
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double* val = &m->data[i][j];
            switch (activation_type) {
                case SIGMOID: *val = sigmoid_derivative(*val); break;
                case RELU: *val = relu_derivative(*val); break;
                case LEAKY_RELU: *val = leaky_relu_derivative(*val); break;
            }
        }
    }
    gann_set_error(GANN_SUCCESS);
}

NeuralNetwork* nn_create(int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output) {
    if (num_layers < 2) {
        gann_set_error(GANN_ERROR_INVALID_ARCHITECTURE);
        return NULL;
    }
    if (!architecture) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }

    NeuralNetwork* net = (NeuralNetwork*)calloc(1, sizeof(NeuralNetwork));
    if (!net) {
        gann_set_error(GANN_ERROR_ALLOC_FAILED);
        return NULL;
    }

    net->num_layers = num_layers;
    net->activation_hidden = activation_hidden;
    net->activation_output = activation_output;

    net->architecture = (int*)malloc(num_layers * sizeof(int));
    if (!net->architecture) {
        free(net);
        gann_set_error(GANN_ERROR_ALLOC_FAILED);
        return NULL;
    }
    memcpy(net->architecture, architecture, num_layers * sizeof(int));

    int num_weight_sets = num_layers - 1;
    net->weights = (Matrix**)calloc(num_weight_sets, sizeof(Matrix*));
    net->biases = (Matrix**)calloc(num_weight_sets, sizeof(Matrix*));
    net->m_weights = (Matrix**)calloc(num_weight_sets, sizeof(Matrix*));
    net->v_weights = (Matrix**)calloc(num_weight_sets, sizeof(Matrix*));
    net->m_biases = (Matrix**)calloc(num_weight_sets, sizeof(Matrix*));
    net->v_biases = (Matrix**)calloc(num_weight_sets, sizeof(Matrix*));

    if (!net->weights || !net->biases || !net->m_weights || !net->v_weights || !net->m_biases || !net->v_biases) {
        nn_free(net);
        gann_set_error(GANN_ERROR_ALLOC_FAILED);
        return NULL;
    }

    for (int i = 0; i < num_weight_sets; i++) {
        net->weights[i] = create_matrix(architecture[i], architecture[i+1]);
        net->biases[i] = create_matrix(1, architecture[i+1]);
        net->m_weights[i] = create_matrix(architecture[i], architecture[i+1]);
        net->v_weights[i] = create_matrix(architecture[i], architecture[i+1]);
        net->m_biases[i] = create_matrix(1, architecture[i+1]);
        net->v_biases[i] = create_matrix(1, architecture[i+1]);
        if (!net->weights[i] || !net->biases[i] || !net->m_weights[i] || !net->v_weights[i] || !net->m_biases[i] || !net->v_biases[i]) {
            nn_free(net);
            // create_matrix already sets the error code
            return NULL;
        }
    }
    gann_set_error(GANN_SUCCESS);
    return net;
}

void nn_init(NeuralNetwork* net) {
    if (!net) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return;
    }
    for (int i = 0; i < net->num_layers - 1; i++) {
        double limit = sqrt(6.0 / (net->architecture[i] + net->architecture[i+1]));
        for (int r = 0; r < net->weights[i]->rows; r++) {
            for (int c = 0; c < net->weights[i]->cols; c++) {
                net->weights[i]->data[r][c] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
            }
        }
    }
    gann_set_error(GANN_SUCCESS);
}

void nn_free(NeuralNetwork* net) {
    if (!net) return;
    if (net->architecture) free(net->architecture);
    int num_weight_sets = net->num_layers > 1 ? net->num_layers - 1 : 0;
    if (net->weights) {
        for (int i = 0; i < num_weight_sets; i++) free_matrix(net->weights[i]);
        free(net->weights);
    }
    if (net->biases) {
        for (int i = 0; i < num_weight_sets; i++) free_matrix(net->biases[i]);
        free(net->biases);
    }
    if (net->m_weights) {
        for (int i = 0; i < num_weight_sets; i++) free_matrix(net->m_weights[i]);
        free(net->m_weights);
    }
    if (net->v_weights) {
        for (int i = 0; i < num_weight_sets; i++) free_matrix(net->v_weights[i]);
        free(net->v_weights);
    }
    if (net->m_biases) {
        for (int i = 0; i < num_weight_sets; i++) free_matrix(net->m_biases[i]);
        free(net->m_biases);
    }
    if (net->v_biases) {
        for (int i = 0; i < num_weight_sets; i++) free_matrix(net->v_biases[i]);
        free(net->v_biases);
    }
    free(net);
}

Matrix* nn_forward_pass(const NeuralNetwork* net, const Matrix* input) {
    if (!net || !input) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    if (input->cols != net->architecture[0]) {
        gann_set_error(GANN_ERROR_INVALID_DIMENSIONS);
        return NULL;
    }

    Matrix* current_output = matrix_copy(input);
    if (!current_output) return NULL; // matrix_copy sets the error

    for (int i = 0; i < net->num_layers - 1; i++) {
        Matrix* weighted_sum = dot_product(current_output, net->weights[i]);
        if (!weighted_sum) {
            free_matrix(current_output);
            return NULL; // dot_product sets the error
        }
        free_matrix(current_output); // Free previous step's output

        add_bias(weighted_sum, net->biases[i]);
        // add_bias doesn't return a value, but can set an error.
        if (gann_get_last_error() != GANN_SUCCESS) {
            free_matrix(weighted_sum);
            return NULL;
        }

        ActivationType activation = (i < net->num_layers - 2) ? net->activation_hidden : net->activation_output;
        nn_apply_activation(weighted_sum, activation);

        current_output = weighted_sum;
    }
    gann_set_error(GANN_SUCCESS);
    return current_output;
}

NeuralNetwork* nn_clone(const NeuralNetwork* src_net) {
    if (!src_net) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }

    NeuralNetwork* new_net = nn_create(src_net->num_layers, src_net->architecture, src_net->activation_hidden, src_net->activation_output);
    if (!new_net) return NULL; // nn_create sets the error

    for (int i = 0; i < src_net->num_layers - 1; i++) {
        for (int r = 0; r < src_net->weights[i]->rows; r++) {
            for (int c = 0; c < src_net->weights[i]->cols; c++) {
                new_net->weights[i]->data[r][c] = src_net->weights[i]->data[r][c];
            }
        }
        for (int c = 0; c < src_net->biases[i]->cols; c++) {
            new_net->biases[i]->data[0][c] = src_net->biases[i]->data[0][c];
        }
    }
    gann_set_error(GANN_SUCCESS);
    return new_net;
}

int nn_save(const NeuralNetwork* net, const char* filepath) {
    if (!net || !filepath) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return 0;
    }
    FILE* file = fopen(filepath, "wb");
    if (!file) {
        gann_set_error(GANN_ERROR_FILE_OPEN);
        return 0;
    }

    // Macro to handle write errors
#define CHECK_WRITE(data, size, count, file_ptr) \
    if (fwrite(data, size, count, file_ptr) != count) { \
        gann_set_error(GANN_ERROR_FILE_WRITE); \
        fclose(file_ptr); \
        return 0; \
    }

    // Write header: num_layers, activation_hidden, activation_output
    CHECK_WRITE(&net->num_layers, sizeof(int), 1, file);
    CHECK_WRITE(&net->activation_hidden, sizeof(ActivationType), 1, file);
    CHECK_WRITE(&net->activation_output, sizeof(ActivationType), 1, file);

    // Write architecture
    CHECK_WRITE(net->architecture, sizeof(int), net->num_layers, file);

    // Write weights and biases
    for (int i = 0; i < net->num_layers - 1; i++) {
        for (int j = 0; j < net->weights[i]->rows; j++) {
            CHECK_WRITE(net->weights[i]->data[j], sizeof(double), net->weights[i]->cols, file);
        }
        CHECK_WRITE(net->biases[i]->data[0], sizeof(double), net->biases[i]->cols, file);
    }

#undef CHECK_WRITE
    fclose(file);
    gann_set_error(GANN_SUCCESS);
    return 1;
}

NeuralNetwork* nn_load(const char* filepath) {
    if (!filepath) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        gann_set_error(GANN_ERROR_FILE_OPEN);
        return NULL;
    }

    // Macro to handle read errors
#define CHECK_READ(data, size, count, file_ptr) \
    if (fread(data, size, count, file_ptr) != count) { \
        gann_set_error(GANN_ERROR_FILE_READ); \
        fclose(file_ptr); \
        return NULL; \
    }

    int num_layers;
    ActivationType activation_hidden, activation_output;

    CHECK_READ(&num_layers, sizeof(int), 1, file);
    CHECK_READ(&activation_hidden, sizeof(ActivationType), 1, file);
    CHECK_READ(&activation_output, sizeof(ActivationType), 1, file);

    if (num_layers < 2) {
        gann_set_error(GANN_ERROR_INVALID_FILE_FORMAT);
        fclose(file);
        return NULL;
    }

    int* architecture = (int*)malloc(num_layers * sizeof(int));
    if (!architecture) {
        gann_set_error(GANN_ERROR_ALLOC_FAILED);
        fclose(file);
        return NULL;
    }
    if (fread(architecture, sizeof(int), num_layers, file) != num_layers) {
        gann_set_error(GANN_ERROR_FILE_READ);
        free(architecture);
        fclose(file);
        return NULL;
    }

    NeuralNetwork* net = nn_create(num_layers, architecture, activation_hidden, activation_output);
    free(architecture);
    if (!net) {
        // nn_create sets the error
        fclose(file);
        return NULL;
    }

    // Read weights and biases
    for (int i = 0; i < net->num_layers - 1; i++) {
        for (int j = 0; j < net->weights[i]->rows; j++) {
            if (fread(net->weights[i]->data[j], sizeof(double), net->weights[i]->cols, file) != net->weights[i]->cols) {
                gann_set_error(GANN_ERROR_FILE_READ);
                nn_free(net);
                fclose(file);
                return NULL;
            }
        }
        if (fread(net->biases[i]->data[0], sizeof(double), net->biases[i]->cols, file) != net->biases[i]->cols) {
            gann_set_error(GANN_ERROR_FILE_READ);
            nn_free(net);
            fclose(file);
            return NULL;
        }
    }

#undef CHECK_READ
    fclose(file);
    gann_set_error(GANN_SUCCESS);
    return net;
}
