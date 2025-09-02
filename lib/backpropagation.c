#include "backpropagation.h"
#include "matrix.h"
#include "neural_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Main function to train the network using backpropagation
void backpropagate(NeuralNetwork* net, const Dataset* train_dataset, double learning_rate, int epochs, int batch_size) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Here we would shuffle the dataset for better training, but for simplicity, we'll iterate sequentially.

        for (int i = 0; i < train_dataset->num_items; i += batch_size) {
            int current_batch_size = (i + batch_size > train_dataset->num_items) ? (train_dataset->num_items - i) : batch_size;

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
                    apply_activation(z, activation_type);
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
                    apply_activation_derivative(z_derivative, net->activation_hidden);

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
            double lr_batch = learning_rate / current_batch_size;
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

            // --- 4. Free Gradient Accumulators ---
            for (int l = 0; l < net->num_layers - 1; l++) {
                free_matrix(weight_gradients[l]);
                free_matrix(bias_gradients[l]);
            }
            free(weight_gradients);
            free(bias_gradients);
        }
        printf("Epoch %d/%d completed.\n", epoch + 1, epochs);
    }
}
