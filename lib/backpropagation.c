#include "backpropagation.h"
#include "matrix.h"
#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "gann.h"
#include <math.h>

// --- Optimizer-specific Weight Update Functions ---

void update_weights_sgd(NeuralNetwork* net, Matrix** weight_gradients, Matrix** bias_gradients, const GannBackpropParams* params, int batch_size) {
    if (net == NULL || weight_gradients == NULL || bias_gradients == NULL || params == NULL) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return;
    }
    double lr_batch = params->learning_rate / batch_size;
    for (int l = 0; l < net->num_layers - 1; l++) {
        // Update weights
        for(int r=0; r < net->weights[l]->rows; r++) {
            for (int c=0; c < net->weights[l]->cols; c++) {
                net->weights[l]->data[r][c] -= lr_batch * weight_gradients[l]->data[r][c];
            }
        }
        // Update biases
         for (int c=0; c < net->biases[l]->cols; c++) {
            net->biases[l]->data[0][c] -= lr_batch * bias_gradients[l]->data[0][c];
        }
    }
}

void update_weights_rmsprop(NeuralNetwork* net, Matrix** weight_gradients, Matrix** bias_gradients, const GannBackpropParams* params, int batch_size) {
    if (net == NULL || !net->optimizer_state || weight_gradients == NULL || bias_gradients == NULL || params == NULL) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return;
    }
    double lr = params->learning_rate;
    double beta2 = params->beta2;
    double epsilon = params->epsilon;
    OptimizerState* opt_state = net->optimizer_state;

    for (int l = 0; l < net->num_layers - 1; l++) {
        // Update weights
        for (int r = 0; r < net->weights[l]->rows; r++) {
            for (int c = 0; c < net->weights[l]->cols; c++) {
                double grad = weight_gradients[l]->data[r][c] / batch_size;
                opt_state->v_weights[l]->data[r][c] = beta2 * opt_state->v_weights[l]->data[r][c] + (1 - beta2) * (grad * grad);
                net->weights[l]->data[r][c] -= (lr / (sqrt(opt_state->v_weights[l]->data[r][c]) + epsilon)) * grad;
            }
        }
        // Update biases
        for (int c = 0; c < net->biases[l]->cols; c++) {
            double grad = bias_gradients[l]->data[0][c] / batch_size;
            opt_state->v_biases[l]->data[0][c] = beta2 * opt_state->v_biases[l]->data[0][c] + (1 - beta2) * (grad * grad);
            net->biases[l]->data[0][c] -= (lr / (sqrt(opt_state->v_biases[l]->data[0][c]) + epsilon)) * grad;
        }
    }
}

void update_weights_adam(NeuralNetwork* net, Matrix** weight_gradients, Matrix** bias_gradients, const GannBackpropParams* params, int batch_size, int t) {
    if (net == NULL || !net->optimizer_state || weight_gradients == NULL || bias_gradients == NULL || params == NULL) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return;
    }
    double lr = params->learning_rate;
    double beta1 = params->beta1;
    double beta2 = params->beta2;
    double epsilon = params->epsilon;
    OptimizerState* opt_state = net->optimizer_state;

    for (int l = 0; l < net->num_layers - 1; l++) {
        // Update weights
        for (int r = 0; r < net->weights[l]->rows; r++) {
            for (int c = 0; c < net->weights[l]->cols; c++) {
                double grad = weight_gradients[l]->data[r][c] / batch_size;
                // Update moments
                opt_state->m_weights[l]->data[r][c] = beta1 * opt_state->m_weights[l]->data[r][c] + (1 - beta1) * grad;
                opt_state->v_weights[l]->data[r][c] = beta2 * opt_state->v_weights[l]->data[r][c] + (1 - beta2) * (grad * grad);
                // Bias correction
                double m_hat = opt_state->m_weights[l]->data[r][c] / (1 - pow(beta1, t));
                double v_hat = opt_state->v_weights[l]->data[r][c] / (1 - pow(beta2, t));
                // Update weights
                net->weights[l]->data[r][c] -= (lr * m_hat) / (sqrt(v_hat) + epsilon);
            }
        }
        // Update biases
        for (int c = 0; c < net->biases[l]->cols; c++) {
            double grad = bias_gradients[l]->data[0][c] / batch_size;
            // Update moments
            opt_state->m_biases[l]->data[0][c] = beta1 * opt_state->m_biases[l]->data[0][c] + (1 - beta1) * grad;
            opt_state->v_biases[l]->data[0][c] = beta2 * opt_state->v_biases[l]->data[0][c] + (1 - beta2) * (grad * grad);
            // Bias correction
            double m_hat = opt_state->m_biases[l]->data[0][c] / (1 - pow(beta1, t));
            double v_hat = opt_state->v_biases[l]->data[0][c] / (1 - pow(beta2, t));
            // Update biases
            net->biases[l]->data[0][c] -= (lr * m_hat) / (sqrt(v_hat) + epsilon);
        }
    }
}


// --- Utility function to calculate Mean Squared Error ---
double calculate_mse(const NeuralNetwork* net, const Dataset* dataset) {
    if (net == NULL || dataset == NULL || dataset->num_items == 0) {
        return -1.0; // Indicate error
    }

    double total_mse = 0.0;
    for (int i = 0; i < dataset->num_items; i++) {
        Matrix* input = matrix_get_row(dataset->images, i);
        Matrix* target = matrix_get_row(dataset->labels, i);
        Matrix* output = nn_forward_pass(net, input);

        if (output == NULL || target == NULL) {
            if(input) free_matrix(input);
            if(target) free_matrix(target);
            if(output) free_matrix(output);
            continue; // Skip if there was an error
        }

        Matrix* error = matrix_subtract(output, target);
        if (error == NULL) {
            free_matrix(input);
            free_matrix(target);
            free_matrix(output);
            if(error) free_matrix(error);
            continue;
        }

        double mse = 0.0;
        for (int j = 0; j < error->cols; j++) {
            mse += error->data[0][j] * error->data[0][j];
        }
        total_mse += mse / error->cols;

        free_matrix(input);
        free_matrix(target);
        free_matrix(output);
        free_matrix(error);
    }

    return total_mse / dataset->num_items;
}

// --- Private Helper Functions for `backpropagate` ---

/**
 * @brief Allocates matrices to store accumulated gradients for a batch.
 */
static int create_gradient_accumulators(NeuralNetwork* net, Matrix*** out_wg, Matrix*** out_bg) {
    int num_layers = net->num_layers;
    Matrix** wg = calloc(num_layers - 1, sizeof(Matrix*));
    Matrix** bg = calloc(num_layers - 1, sizeof(Matrix*));

    if (!wg || !bg) {
        free(wg);
        free(bg);
        gann_set_error(GANN_ERROR_ALLOC_FAILED);
        return 0;
    }

    for (int l = 0; l < num_layers - 1; l++) {
        wg[l] = create_matrix(net->weights[l]->rows, net->weights[l]->cols);
        bg[l] = create_matrix(net->biases[l]->rows, net->biases[l]->cols);
        if (!wg[l] || !bg[l]) {
            for (int i = 0; i < l; i++) { // Clean up previously allocated matrices
                free_matrix(wg[i]);
                free_matrix(bg[i]);
            }
            free(wg);
            free(bg);
            // create_matrix sets the error
            return 0;
        }
    }
    *out_wg = wg;
    *out_bg = bg;
    return 1;
}

/**
 * @brief Frees the gradient accumulator matrices.
 */
static void free_gradient_accumulators(Matrix** wg, Matrix** bg, int num_layers) {
    if (wg) {
        for (int l = 0; l < num_layers - 1; l++) free_matrix(wg[l]);
        free(wg);
    }
    if (bg) {
        for (int l = 0; l < num_layers - 1; l++) free_matrix(bg[l]);
        free(bg);
    }
}

/**
 * @brief Performs a forward pass, storing all intermediate activations and z-values.
 */
static int forward_pass_and_store(const NeuralNetwork* net, const Matrix* input, Matrix*** out_activations, Matrix*** out_z_values) {
    Matrix** activations = calloc(net->num_layers, sizeof(Matrix*));
    Matrix** z_values = calloc(net->num_layers - 1, sizeof(Matrix*));
    if (!activations || !z_values) {
        free(activations);
        free(z_values);
        gann_set_error(GANN_ERROR_ALLOC_FAILED);
        return 0;
    }

    activations[0] = matrix_copy(input);
    if (!activations[0]) goto error;

    for (int l = 0; l < net->num_layers - 1; l++) {
        Matrix* z = dot_product(activations[l], net->weights[l]);
        if (!z) goto error;
        add_bias(z, net->biases[l]);
        z_values[l] = matrix_copy(z);
        if (!z_values[l]) { free_matrix(z); goto error; }

        ActivationType activation_type = (l == net->num_layers - 2) ? net->activation_output : net->activation_hidden;
        nn_apply_activation(z, activation_type);
        activations[l + 1] = z;
    }

    *out_activations = activations;
    *out_z_values = z_values;
    return 1;

error:
    for (int i = 0; i < net->num_layers; i++) free_matrix(activations[i]);
    for (int i = 0; i < net->num_layers - 1; i++) free_matrix(z_values[i]);
    free(activations);
    free(z_values);
    return 0;
}

/**
 * @brief Performs the backward pass to calculate and accumulate gradients for one sample.
 */
static int backward_pass_and_accumulate(const NeuralNetwork* net, const Matrix* target, Matrix** activations, Matrix** z_values, Matrix** weight_gradients, Matrix** bias_gradients) {
    Matrix *delta = NULL, *activations_T = NULL, *dw = NULL;
    int success = 0;

    // Calculate delta for the output layer: (y_pred - y_true)
    delta = matrix_subtract(activations[net->num_layers - 1], target);
    if (!delta) goto cleanup;

    // --- Calculate gradients for the last layer ---
    activations_T = matrix_transpose(activations[net->num_layers - 2]);
    if (!activations_T) goto cleanup;

    dw = dot_product(activations_T, delta);
    if (!dw) goto cleanup;

    // Accumulate gradients
    for (int r = 0; r < dw->rows; r++) for (int c = 0; c < dw->cols; c++) weight_gradients[net->num_layers - 2]->data[r][c] += dw->data[r][c];
    for (int c = 0; c < delta->cols; c++) bias_gradients[net->num_layers - 2]->data[0][c] += delta->data[0][c];
    free_matrix(dw); dw = NULL;
    free_matrix(activations_T); activations_T = NULL;

    // --- Propagate error backward ---
    for (int l = net->num_layers - 3; l >= 0; l--) {
        Matrix* weights_T = matrix_transpose(net->weights[l + 1]);
        Matrix* next_delta = dot_product(delta, weights_T);
        free_matrix(delta); delta = NULL;
        free_matrix(weights_T);
        if (!next_delta) goto cleanup;

        Matrix* z_derivative = matrix_copy(z_values[l]);
        if (!z_derivative) { free_matrix(next_delta); goto cleanup; }
        nn_apply_activation_derivative(z_derivative, net->activation_hidden);

        delta = matrix_elementwise_multiply(next_delta, z_derivative);
        free_matrix(next_delta);
        free_matrix(z_derivative);
        if (!delta) goto cleanup;

        // Calculate and accumulate gradients for the current layer
        activations_T = matrix_transpose(activations[l]);
        if (!activations_T) goto cleanup;
        dw = dot_product(activations_T, delta);
        if (!dw) goto cleanup;

        for (int r = 0; r < dw->rows; r++) for (int c = 0; c < dw->cols; c++) weight_gradients[l]->data[r][c] += dw->data[r][c];
        for (int c = 0; c < delta->cols; c++) bias_gradients[l]->data[0][c] += delta->data[0][c];
        free_matrix(dw); dw = NULL;
        free_matrix(activations_T); activations_T = NULL;
    }
    success = 1;

cleanup:
    free_matrix(delta);
    free_matrix(activations_T);
    free_matrix(dw);
    return success;
}

// Main function to train the network using backpropagation
void backpropagate(NeuralNetwork* net, const Dataset* train_dataset, const GannBackpropParams* params, const Dataset* validation_dataset) {
    if (net == NULL || train_dataset == NULL || params == NULL) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return;
    }

    double best_validation_accuracy = -1.0;
    int epochs_without_improvement = 0;
    NeuralNetwork* best_network_state = NULL;
    int t = 0; // Timestep for Adam

    for (int epoch = 0; epoch < params->epochs; epoch++) {
        for (int i = 0; i < train_dataset->num_items; i += params->batch_size) {
            t++;
            int current_batch_size = (i + params->batch_size > train_dataset->num_items) ? (train_dataset->num_items - i) : params->batch_size;
            Matrix **weight_gradients = NULL, **bias_gradients = NULL;

            if (!create_gradient_accumulators(net, &weight_gradients, &bias_gradients)) {
                goto end_training; // Critical error
            }

            for (int j = 0; j < current_batch_size; j++) {
                Matrix *input = NULL, *target = NULL;
                Matrix **activations = NULL, **z_values = NULL;

                input = matrix_get_row(train_dataset->images, i + j);
                target = matrix_get_row(train_dataset->labels, i + j);
                if (!input || !target) { free_matrix(input); free_matrix(target); continue; }

                if (!forward_pass_and_store(net, input, &activations, &z_values)) {
                    free_matrix(input); free_matrix(target); continue;
                }

                backward_pass_and_accumulate(net, target, activations, z_values, weight_gradients, bias_gradients);

                // Free memory for this sample
                free_matrix(input);
                free_matrix(target);
                for(int l=0; l<net->num_layers; l++) free_matrix(activations[l]);
                for(int l=0; l<net->num_layers-1; l++) free_matrix(z_values[l]);
                free(activations);
                free(z_values);
            }

            // Update weights
            switch (params->optimizer_type) {
                case ADAM: update_weights_adam(net, weight_gradients, bias_gradients, params, current_batch_size, t); break;
                case RMSPROP: update_weights_rmsprop(net, weight_gradients, bias_gradients, params, current_batch_size); break;
                default: update_weights_sgd(net, weight_gradients, bias_gradients, params, current_batch_size); break;
            }
            free_gradient_accumulators(weight_gradients, bias_gradients, net->num_layers);
        }

        if (params->logging) {
            double train_accuracy = gann_evaluate(net, train_dataset);
            printf("Epoch %d/%d, Train Accuracy: %.2f%%\n", epoch + 1, params->epochs, train_accuracy * 100.0);
        }

        if (validation_dataset && params->early_stopping_patience > 0) {
            double val_acc = gann_evaluate(net, validation_dataset);
            if (params->logging) printf("  Validation Accuracy: %.2f%%\n", val_acc * 100.0);
            if (val_acc > best_validation_accuracy + params->early_stopping_threshold) {
                best_validation_accuracy = val_acc;
                epochs_without_improvement = 0;
                if (best_network_state) nn_free(best_network_state);
                best_network_state = nn_clone(net);
            } else {
                epochs_without_improvement++;
            }
            if (epochs_without_improvement >= params->early_stopping_patience) {
                if (params->logging) printf("Early stopping triggered after %d epochs without improvement.\n", params->early_stopping_patience);
                goto end_training;
            }
        }
    }

end_training:
    if (best_network_state) {
        for (int l = 0; l < net->num_layers - 1; l++) {
            matrix_copy_data(net->weights[l], best_network_state->weights[l]);
            matrix_copy_data(net->biases[l], best_network_state->biases[l]);
        }
        nn_free(best_network_state);
    }
}
