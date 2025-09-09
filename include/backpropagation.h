#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "neural_network.h"
#include "data_loader.h"
#include <stdbool.h>

/**
 * @brief Enumeration of supported optimization algorithms for backpropagation.
 */
typedef enum {
    SGD,     /**< Stochastic Gradient Descent. */
    ADAM,    /**< Adam optimizer, which adapts learning rates. */
    RMSPROP  /**< RMSprop optimizer. */
} OptimizerType;

/**
 * @brief Parameters for training a neural network with backpropagation.
 * @details This struct holds all the parameters needed to configure the
 * backpropagation training process, including network architecture, learning
 * parameters, and choice of optimizer.
 */
typedef struct {
    const int* architecture;        /**< An array defining the number of neurons in each layer, e.g., `{784, 128, 10}`. */
    int num_layers;                 /**< The total number of layers in the network (size of the `architecture` array). */
    double learning_rate;           /**< The step size for updating weights during gradient descent. */
    int epochs;                     /**< The number of times the training algorithm will iterate over the entire dataset. */
    int batch_size;                 /**< The number of training samples to process before making a weight update. */
    ActivationType activation_hidden; /**< The activation function to use for all hidden layers (e.g., `RELU`). */
    ActivationType activation_output; /**< The activation function to use for the output layer (e.g., `SIGMOID`). */
    OptimizerType optimizer_type;   /**< The optimization algorithm to use (e.g., `ADAM`, `SGD`). */
    double beta1;                   /**< The exponential decay rate for the first moment estimates. Used by the Adam optimizer. Default is 0.9. */
    double beta2;                   /**< The exponential decay rate for the second-moment estimates. Used by Adam and RMSprop. Default is 0.999. */
    double epsilon;                 /**< A small constant added for numerical stability. Used by Adam and RMSprop. Default is 1e-8. */
    bool logging;                   /**< If true, prints progress information (epoch, accuracy) to the console during training. */
    int early_stopping_patience;    /**< Number of epochs with no improvement on the validation set to wait before stopping. 0 to disable. */
    double early_stopping_threshold;/**< The minimum improvement in validation accuracy required to reset the patience counter. */
} GannBackpropParams;


/**
 * @brief Trains a neural network using the backpropagation algorithm.
 * @details This is the core function for backpropagation training. It iterates
 * over the dataset for a specified number of epochs, processing the data in
 * batches. In each batch, it computes gradients and updates the network's
 * weights and biases using the chosen optimizer.
 * @param net The neural network to be trained (will be modified in place).
 * @param train_dataset The dataset used for training.
 * @param params The parameters for the backpropagation algorithm, including learning rate, epochs, etc.
 * @param validation_dataset An optional dataset for validation, used for logging and early stopping. Can be `NULL`.
 */
void backpropagate(NeuralNetwork* net, const Dataset* train_dataset, const GannBackpropParams* params, const Dataset* validation_dataset);

/**
 * @brief Updates network weights using Stochastic Gradient Descent (SGD).
 * @note This function is exposed primarily for testing purposes.
 * @param net The neural network to update.
 * @param weight_gradients An array of matrices containing the accumulated gradients for the weights.
 * @param bias_gradients An array of matrices containing the accumulated gradients for the biases.
 * @param params The backpropagation parameters, used to get the learning rate.
 * @param batch_size The number of samples in the batch that the gradients were accumulated over.
 */
void update_weights_sgd(NeuralNetwork* net, Matrix** weight_gradients, Matrix** bias_gradients, const GannBackpropParams* params, int batch_size);

/**
 * @brief Updates network weights using the RMSprop algorithm.
 * @note This function is exposed primarily for testing purposes.
 * @param net The neural network to update. It must have its optimizer state initialized.
 * @param weight_gradients An array of matrices containing the accumulated gradients for the weights.
 * @param bias_gradients An array of matrices containing the accumulated gradients for the biases.
 * @param params The backpropagation parameters, used to get learning rate, beta2, and epsilon.
 * @param batch_size The number of samples in the batch.
 */
void update_weights_rmsprop(NeuralNetwork* net, Matrix** weight_gradients, Matrix** bias_gradients, const GannBackpropParams* params, int batch_size);

/**
 * @brief Updates network weights using the Adam algorithm.
 * @note This function is exposed primarily for testing purposes.
 * @param net The neural network to update. It must have its optimizer state initialized.
 * @param weight_gradients An array of matrices containing the accumulated gradients for the weights.
 * @param bias_gradients An array of matrices containing the accumulated gradients for the biases.
 * @param params The backpropagation parameters, used to get learning rate and other Adam-specific hyperparameters.
 * @param batch_size The number of samples in the batch.
 * @param t The current timestep, used for bias correction in Adam.
 */
void update_weights_adam(NeuralNetwork* net, Matrix** weight_gradients, Matrix** bias_gradients, const GannBackpropParams* params, int batch_size, int t);

/**
 * @brief Calculates the mean squared error (MSE) for a network on a given dataset.
 * @note This function is exposed primarily for testing and validation purposes.
 * @param net The neural network to evaluate.
 * @param dataset The dataset to evaluate the network on.
 * @return The average mean squared error across all items in the dataset. Returns -1.0 on error.
 */
double calculate_mse(const NeuralNetwork* net, const Dataset* dataset);


#endif // BACKPROPAGATION_H
