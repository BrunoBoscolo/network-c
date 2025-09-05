#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "neural_network.h"
#include "data_loader.h"

/**
 * @brief Enum for optimizer types.
 */
typedef enum {
    SGD,
    ADAM,
    RMSPROP
} OptimizerType;

/**
 * @brief Parameters for the gann_train_with_backprop function.
 */
typedef struct {
    const int* architecture;        /**< An array defining the number of neurons in each layer. */
    int num_layers;                 /**< The total number of layers in the network. */
    double learning_rate;           /**< The step size for gradient descent. */
    int epochs;                     /**< The number of times to iterate over the entire dataset. */
    int batch_size;                 /**< The number of samples to process before updating weights. */
    ActivationType activation_hidden; /**< The activation function to use for the hidden layers. */
    ActivationType activation_output; /**< The activation function to use for the output layer. */
    OptimizerType optimizer_type;   /**< The optimization algorithm to use. */
    double beta1;                   /**< The exponential decay rate for the first moment estimates (for Adam). */
    double beta2;                   /**< The exponential decay rate for the second-moment estimates (for Adam and RMSprop). */
    double epsilon;                 /**< A small constant for numerical stability (for Adam and RMSprop). */
} GannBackpropParams;


/**
 * @brief Trains a neural network using the backpropagation algorithm.
 *
 * @param net The neural network to be trained.
 * @param train_dataset The dataset used for training.
 * @param params The parameters for the backpropagation algorithm.
 */
void backpropagate(NeuralNetwork* net, const Dataset* train_dataset, const GannBackpropParams* params);

// --- Optimizer-specific Weight Update Functions (exposed for testing) ---
void update_weights_rmsprop(NeuralNetwork* net, Matrix** weight_gradients, Matrix** bias_gradients, const GannBackpropParams* params, int batch_size);
void update_weights_adam(NeuralNetwork* net, Matrix** weight_gradients, Matrix** bias_gradients, const GannBackpropParams* params, int batch_size, int t);


#endif // BACKPROPAGATION_H
