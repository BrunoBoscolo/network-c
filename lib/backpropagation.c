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

// Main function to train the network using backpropagation
void backpropagate(NeuralNetwork* net, const Dataset* train_dataset, const GannBackpropParams* params) {
    if (net == NULL || train_dataset == NULL || params == NULL) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return;
    }
    int t = 0; // Timestep for Adam
    for (int epoch = 0; epoch < params->epochs; epoch++) {
        // Here we would shuffle the dataset for better training, but for simplicity, we'll iterate sequentially.

        for (int i = 0; i < train_dataset->num_items; i += params->batch_size) {
            t++;
            int current_batch_size = (i + params->batch_size > train_dataset->num_items) ? (train_dataset->num_items - i) : params->batch_size;

            // --- 1. Initialize Gradient Accumulators ---
            Matrix** weight_gradients = malloc((net->num_layers - 1) * sizeof(Matrix*));
            Matrix** bias_gradients = malloc((net->num_layers - 1) * sizeof(Matrix*));
            for (int l = 0; l < net->num_layers - 1; l++) {
                weight_gradients[l] = create_matrix(net->weights[l]->rows, net->weights[l]->cols);
                bias_gradients[l] = create_matrix(net->biases[l]->rows, net->biases[l]->cols);
            }

            // --- 2. Process Batch ---
            for (int j = 0; j < current_batch_size; j++) {
                int sample_idx = i + j;

                // --- a. Forward Pass ---
                Matrix* input = matrix_get_row(train_dataset->images, sample_idx);

                // Store activations and pre-activations (z-values) for each layer
                Matrix** activations = malloc(net->num_layers * sizeof(Matrix*));
                Matrix** z_values = malloc((net->num_layers - 1) * sizeof(Matrix*));
                activations[0] = matrix_copy(input);

                for (int l = 0; l < net->num_layers - 1; l++) {
                    Matrix* z = dot_product(activations[l], net->weights[l]);
                    add_bias(z, net->biases[l]);
                    z_values[l] = matrix_copy(z);

                    ActivationType activation_type = (l == net->num_layers - 2) ? net->activation_output : net->activation_hidden;
                    nn_apply_activation(z, activation_type);
                    activations[l + 1] = z; // z is now the activation
                }

                // --- b. Backward Pass ---

                // Calculate output error (delta)
                Matrix* target = matrix_get_row(train_dataset->labels, sample_idx);

                Matrix* output_error = matrix_subtract(activations[net->num_layers - 1], target); // (y - y_hat)

                // Calculate delta for the output layer
                // For MSE, delta = (y_pred - y_true) * activation_derivative(z)
                // For cross-entropy with sigmoid, delta is just (y_pred - y_true)
                // Using the simpler form is more stable.
                Matrix* delta = matrix_copy(output_error);


                // --- c. Calculate Gradients for the last layer ---
                Matrix* activations_T = matrix_transpose(activations[net->num_layers - 2]);
                Matrix* dw = dot_product(activations_T, delta);

                // Accumulate gradients
                for(int r=0; r<dw->rows; r++) for(int c=0; c<dw->cols; c++) weight_gradients[net->num_layers-2]->data[r][c] += dw->data[r][c];
                for(int c=0; c<delta->cols; c++) bias_gradients[net->num_layers-2]->data[0][c] += delta->data[0][c];

                free_matrix(dw);
                free_matrix(activations_T);

                // --- d. Propagate error backward ---
                for (int l = net->num_layers - 3; l >= 0; l--) {
                    Matrix* weights_T = matrix_transpose(net->weights[l + 1]);
                    Matrix* next_delta = dot_product(delta, weights_T);
                    free_matrix(delta); // Free old delta
                    free_matrix(weights_T);

                    Matrix* z_derivative = matrix_copy(z_values[l]);
                    nn_apply_activation_derivative(z_derivative, net->activation_hidden);

                    delta = matrix_elementwise_multiply(next_delta, z_derivative);
                    free_matrix(next_delta);
                    free_matrix(z_derivative);

                    // Calculate gradients for the current layer
                    activations_T = matrix_transpose(activations[l]);
                    dw = dot_product(activations_T, delta);

                    // Accumulate gradients
                    for(int r=0; r<dw->rows; r++) for(int c=0; c<dw->cols; c++) weight_gradients[l]->data[r][c] += dw->data[r][c];
                    for(int c=0; c<delta->cols; c++) bias_gradients[l]->data[0][c] += delta->data[0][c];

                    free_matrix(dw);
                    free_matrix(activations_T);
                }

                // --- e. Free memory for this sample ---
                free_matrix(delta);
                free_matrix(output_error);
                free_matrix(target);
                free_matrix(input);
                for(int l=0; l<net->num_layers; l++) free_matrix(activations[l]);
                for(int l=0; l<net->num_layers-1; l++) free_matrix(z_values[l]);
                free(activations);
                free(z_values);
            }

            // --- 3. Update Weights and Biases ---
            switch (params->optimizer_type) {
                case ADAM:
                    update_weights_adam(net, weight_gradients, bias_gradients, params, current_batch_size, t);
                    break;
                case RMSPROP:
                    update_weights_rmsprop(net, weight_gradients, bias_gradients, params, current_batch_size);
                    break;
                case SGD:
                default:
                    update_weights_sgd(net, weight_gradients, bias_gradients, params, current_batch_size);
                    break;
            }

            // --- 4. Free Gradient Accumulators ---
            for (int l = 0; l < net->num_layers - 1; l++) {
                free_matrix(weight_gradients[l]);
                free_matrix(bias_gradients[l]);
            }
            free(weight_gradients);
            free(bias_gradients);
        }
        if (params->logging) {
            double mse = calculate_mse(net, train_dataset);
            printf("Epoch %d/%d, MSE: %f\n", epoch + 1, params->epochs, mse);
        }
    }
}
