#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "neural_network.h"
#include "data_loader.h"

/**
 * @brief Trains a neural network using the backpropagation algorithm.
 *
 * @param net The neural network to be trained.
 * @param train_dataset The dataset used for training.
 * @param learning_rate The step size for gradient descent.
 * @param epochs The number of times to iterate over the entire dataset.
 * @param batch_size The number of samples to process before updating the weights.
 */
void backpropagate(NeuralNetwork* net, const Dataset* train_dataset, double learning_rate, int epochs, int batch_size);


#endif // BACKPROPAGATION_H
