#include "gann_docs.h"
#include <string.h>

typedef struct {
    const char* function_name;
    const char* documentation;
} FunctionDoc;

static const FunctionDoc function_docs[] = {
    {
        "gann_seed_rng",
        "void gann_seed_rng(unsigned int seed)\n\n"
        "Concept:\n"
        "Seeds the random number generator used by the library. This is crucial for reproducibility. "
        "In machine learning, many processes (like weight initialization and genetic algorithm operators) are stochastic. "
        "By seeding the random number generator, you ensure that the sequence of 'random' numbers is the same every time you run the program, "
        "making experiments and debugging much easier."
    },
    {
        "gann_create_default_params",
        "GannTrainParams gann_create_default_params(void)\n\n"
        "Concept:\n"
        "Creates a struct with sensible default training parameters for the genetic algorithm. "
        "Training a neural network, especially with a genetic algorithm, involves tuning many hyperparameters. "
        "This function provides a reasonable starting point, saving the user from having to specify every single parameter from scratch."
    },
    {
        "gann_evolve",
        "NeuralNetwork* gann_evolve(const GannEvolveParams* params, const Dataset* train_dataset, const Dataset* validation_dataset)\n\n"
        "Concept:\n"
        "Evolves a population of neural networks using custom genetic operators. "
        "This function is for advanced users who want to experiment with their own selection, crossover, or mutation algorithms. "
        "It provides the flexibility to extend the library's capabilities without modifying the core source code."
    },
    {
        "gann_train",
        "NeuralNetwork* gann_train(const GannTrainParams* params, const Dataset* train_dataset, const Dataset* validation_dataset)\n\n"
        "Concept:\n"
        "Trains a neural network using a genetic algorithm with default operators. "
        "This function abstracts away the complexity of the genetic algorithm, allowing the user to train a network with a single function call. "
        "It's the main entry point for training with a genetic algorithm."
    },
    {
        "gann_train_with_backprop",
        "NeuralNetwork* gann_train_with_backprop(const GannBackpropParams* params, const Dataset* train_dataset, const Dataset* validation_dataset)\n\n"
        "Concept:\n"
        "Trains a neural network using the backpropagation algorithm. "
        "Backpropagation is a widely used algorithm for training neural networks. "
        "This function provides a more traditional way to train a network, contrasting with the genetic algorithm approach."
    },
    {
        "gann_predict",
        "int gann_predict(const NeuralNetwork* net, const double* input)\n\n"
        "Concept:\n"
        "Makes a prediction on a single input using a trained network. "
        "Once a network is trained, this function is used to perform inference on new, unseen data. "
        "It performs a forward pass and returns the most likely output."
    },
    {
        "gann_evaluate",
        "double gann_evaluate(const NeuralNetwork* net, const Dataset* dataset)\n\n"
        "Concept:\n"
        "Evaluates the network's accuracy on a given dataset. "
        "This is a crucial step to understand how well the network has learned. "
        "It's used to measure the performance of the model on a test set, which it has not seen during training."
    },
    {
        "nn_create",
        "NeuralNetwork* nn_create(int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output)\n\n"
        "Concept:\n"
        "Creates and allocates a new neural network structure. "
        "This function sets up the basic skeleton of the network, including its layers and the number of neurons in each layer. "
        "It's the first step in building a neural network."
    },
    {
        "nn_init",
        "void nn_init(NeuralNetwork* net)\n\n"
        "Concept:\n"
        "Initializes the weights of a neural network with random values. "
        "Proper weight initialization is critical for training neural networks. "
        "This function uses a standard technique (Xavier/Glorot initialization) to prevent issues like vanishing or exploding gradients."
    },
    {
        "nn_free",
        "void nn_free(NeuralNetwork* net)\n\n"
        "Concept:\n"
        "Frees all memory allocated for a neural network. "
        "In C, memory management is manual. This function is essential to prevent memory leaks by deallocating the network and all its components."
    },
    {
        "nn_forward_pass",
        "Matrix* nn_forward_pass(const NeuralNetwork* net, const Matrix* input)\n\n"
        "Concept:\n"
        "Performs a forward pass through the network to compute an output. "
        "This is the core operation of a neural network, where input data is processed through the layers to produce a prediction. "
        "It's the 'thinking' part of the network."
    },
    {
        "nn_clone",
        "NeuralNetwork* nn_clone(const NeuralNetwork* src_net)\n\n"
        "Concept:\n"
        "Creates a deep copy of a neural network. "
        "This is particularly useful in genetic algorithms, where you need to create copies of parent networks to create offspring. "
        "A deep copy ensures that the new network is completely independent of the original."
    },
    {
        "nn_save",
        "int nn_save(const NeuralNetwork* net, const char* filepath)\n\n"
        "Concept:\n"
        "Saves a neural network's structure and parameters to a file. "
        "Training a neural network can take a long time. This function allows you to save your trained model so you can use it later without having to retrain it."
    },
    {
        "nn_load",
        "NeuralNetwork* nn_load(const char* filepath)\n\n"
        "Concept:\n"
        "Loads a neural network from a file. "
        "This function allows you to load a previously saved model and use it for prediction or further training."
    },
    {
        "backpropagate",
        "void backpropagate(NeuralNetwork* net, const Dataset* train_dataset, const GannBackpropParams* params, const Dataset* validation_dataset)\n\n"
        "Concept:\n"
        "The core function for backpropagation training. It iterates over the dataset for a specified number of epochs, processing the data in batches. In each batch, it computes gradients and updates the network's weights and biases using the chosen optimizer."
    },
    {
        "update_weights_sgd",
        "void update_weights_sgd(NeuralNetwork* net, Matrix** weight_gradients, Matrix** bias_gradients, const GannBackpropParams* params, int batch_size)\n\n"
        "Concept:\n"
        "Updates network weights using Stochastic Gradient Descent (SGD). SGD is the simplest optimizer. It updates the weights in the opposite direction of the gradient, scaled by the learning rate. It is computationally efficient but can be slow to converge."
    },
    {
        "update_weights_rmsprop",
        "void update_weights_rmsprop(NeuralNetwork* net, Matrix** weight_gradients, Matrix** bias_gradients, const GannBackpropParams* params, int batch_size)\n\n"
        "Concept:\n"
        "Updates network weights using the RMSprop algorithm. RMSprop is an adaptive learning rate method that divides the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight. It is effective in non-stationary settings."
    },
    {
        "update_weights_adam",
        "void update_weights_adam(NeuralNetwork* net, Matrix** weight_gradients, Matrix** bias_gradients, const GannBackpropParams* params, int batch_size, int t)\n\n"
        "Concept:\n"
        "Updates network weights using the Adam algorithm. Adam (Adaptive Moment Estimation) is another adaptive learning rate method that computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients. It is one of the most popular and effective optimizers."
    },
    {
        "calculate_mse",
        "double calculate_mse(const NeuralNetwork* net, const Dataset* dataset)\n\n"
        "Concept:\n"
        "Calculates the mean squared error (MSE) for a network on a given dataset. MSE is a common loss function used for regression problems. It measures the average squared difference between the estimated values and the actual value."
    },
    {
        "crossover",
        "NeuralNetwork* crossover(const NeuralNetwork* parent1, const NeuralNetwork* parent2, CrossoverType crossover_type)\n\n"
        "Concept:\n"
        "Performs crossover between two parent networks to create a child network. "
        "Crossover (or recombination) is a genetic operator used to combine the genetic information of two parents to generate new offspring. "
        "It's a key part of how a genetic algorithm explores the solution space, mixing existing solutions to find better ones. "
        "This function acts as a dispatcher, calling the specific crossover method (e.g., uniform, single-point) based on the `crossover_type` parameter."
    },
    {
        "load_mnist_dataset",
        "Dataset* load_mnist_dataset(const char* image_path, const char* label_path)\n\n"
        "Concept:\n"
        "Loads the MNIST dataset from the specified IDX-formatted files. "
        "The MNIST dataset is a standard benchmark in machine learning. This function handles the details of reading the binary IDX file format, normalizing the pixel values, and one-hot encoding the labels, so the user can focus on training the model."
    },
    {
        "create_dummy_dataset",
        "Dataset* create_dummy_dataset(int num_items)\n\n"
        "Concept:\n"
        "Creates a dummy dataset with random values for testing purposes. "
        "This is useful for unit testing and debugging. It allows you to test the data loading and processing pipeline without needing the actual MNIST dataset."
    },
    {
        "create_dummy_dataset_with_label",
        "Dataset* create_dummy_dataset_with_label(int num_items, int label)\n\n"
        "Concept:\n"
        "Creates a dummy dataset with a specific label for all items. "
        "This is useful for testing if a network can overfit to a single class, which is a good way to check if the model is capable of learning at all."
    },
    {
        "split_dataset",
        "void split_dataset(const Dataset* original, int split_size, Dataset* out_dataset_1, Dataset* out_dataset_2)\n\n"
        "Concept:\n"
        "Splits a dataset into two new datasets by copying the data. "
        "In machine learning, it's crucial to evaluate a model on data it hasn't seen during training. This function is used to create a training set and a validation set from a single source dataset, which is a standard practice."
    },
    {
        "free_dataset",
        "void free_dataset(Dataset* dataset)\n\n"
        "Concept:\n"
        "Frees the memory allocated for a dataset. "
        "In C, memory management is manual. This function is essential to prevent memory leaks by deallocating the dataset and all its components."
    },
    {
        "evo_create_initial_population",
        "NeuralNetwork** evo_create_initial_population(int population_size, int num_layers, const int* architecture, ActivationType activation_hidden, ActivationType activation_output)\n\n"
        "Concept:\n"
        "Creates the initial population of random neural networks. "
        "A genetic algorithm starts with a diverse set of random solutions. This function creates that initial set of neural networks, each with a different random initialization of weights."
    },
    {
        "evo_reproduce",
        "NeuralNetwork** evo_reproduce(const NetworkFitness* fittest_networks, int num_fittest, int new_population_size, CrossoverType crossover_type, int tournament_size)\n\n"
        "Concept:\n"
        "Creates a new generation of networks through selection and crossover. "
        "This function implements the 'survival of the fittest' principle. It selects the best-performing networks from the current generation and combines them (using crossover) to create the next generation of networks."
    },
    {
        "gann_get_last_error",
        "GannError gann_get_last_error(void)\n\n"
        "Concept:\n"
        "Gets the last error that occurred on the calling thread. "
        "When a library function fails, this function can be called to retrieve a specific error code, which provides more details about the cause of the failure. This is a common pattern for error handling in C libraries."
    },
    {
        "gann_error_to_string",
        "const char* gann_error_to_string(GannError error_code)\n\n"
        "Concept:\n"
        "Converts a `GannError` code into a human-readable, null-terminated string. "
        "This is useful for logging and debugging, as it allows you to print a descriptive error message instead of just an error code."
    },
    {
        "create_matrix",
        "Matrix* create_matrix(int rows, int cols)\n\n"
        "Concept:\n"
        "Creates a new matrix with all elements initialized to zero. "
        "This is the basic constructor for a matrix. It allocates the necessary memory and initializes the matrix to a known state (all zeros)."
    },
    {
        "free_matrix",
        "void free_matrix(Matrix* m)\n\n"
        "Concept:\n"
        "Frees the memory allocated for a matrix. "
        "In C, memory management is manual. This function is essential to prevent memory leaks by deallocating the matrix and its underlying data."
    },
    {
        "print_matrix",
        "void print_matrix(const Matrix* m)\n\n"
        "Concept:\n"
        "Prints the contents of a matrix to the console. "
        "This is a utility function that is useful for debugging and understanding the state of the network at different stages of training."
    },
    {
        "dot_product",
        "Matrix* dot_product(const Matrix* m1, const Matrix* m2)\n\n"
        "Concept:\n"
        "Computes the dot product of two matrices. "
        "The dot product is a fundamental operation in neural networks. It's used to calculate the weighted sum of inputs at each neuron."
    },
    {
        "add_bias",
        "void add_bias(Matrix* m, const Matrix* bias)\n\n"
        "Concept:\n"
        "Adds a bias vector (a row matrix) to each row of a matrix, in place. "
        "In a neural network, the bias term is added to the weighted sum of inputs. This operation applies that bias to the output of a layer."
    },
    {
        "matrix_transpose",
        "Matrix* matrix_transpose(const Matrix* m)\n\n"
        "Concept:\n"
        "Creates a new matrix that is the transpose of the input matrix. "
        "The transpose of a matrix is used in several parts of the backpropagation algorithm, particularly when calculating weight gradients."
    },
    {
        "matrix_elementwise_multiply",
        "Matrix* matrix_elementwise_multiply(const Matrix* m1, const Matrix* m2)\n\n"
        "Concept:\n"
        "Performs element-wise multiplication (Hadamard product) of two matrices. "
        "This operation is used in the backpropagation algorithm to combine gradients."
    },
    {
        "matrix_subtract",
        "Matrix* matrix_subtract(const Matrix* m1, const Matrix* m2)\n\n"
        "Concept:\n"
        "Subtracts the second matrix from the first, element by element. "
        "This is a basic matrix operation used in various calculations, including the calculation of the error between the network's output and the true labels."
    },
    {
        "matrix_add",
        "Matrix* matrix_add(const Matrix* m1, const Matrix* m2)\n\n"
        "Concept:\n"
        "Adds two matrices, element by element. "
        "This is a basic matrix operation used in various calculations."
    },
    {
        "matrix_scale",
        "Matrix* matrix_scale(const Matrix* m, double scalar)\n\n"
        "Concept:\n"
        "Scales a matrix by multiplying every element by a scalar value. "
        "This is used in backpropagation to scale the gradients by the learning rate."
    },
    {
        "matrix_from_array",
        "Matrix* matrix_from_array(const double* array, int rows, int cols)\n\n"
        "Concept:\n"
        "Creates a matrix from a flat, 1D array of data. "
        "This is a utility function to convert data from a simple array to the matrix format used by the library."
    },
    {
        "matrix_copy",
        "Matrix* matrix_copy(const Matrix* m)\n\n"
        "Concept:\n"
        "Creates a deep copy of a matrix. "
        "This is useful when you need to modify a matrix without changing the original."
    },
    {
        "matrix_get_row",
        "Matrix* matrix_get_row(const Matrix* m, int row)\n\n"
        "Concept:\n"
        "Extracts a single row from a matrix and returns it as a new 1xN matrix. "
        "This is useful for processing data one sample at a time."
    },
    {
        "matrix_copy_data",
        "void matrix_copy_data(Matrix* dest, const Matrix* src)\n\n"
        "Concept:\n"
        "Copies the data from a source matrix to a destination matrix. "
        "This is a more efficient way to copy matrix data than creating a new matrix, as it avoids memory allocation."
    },
    {
        "mutate_network",
        "void mutate_network(NeuralNetwork* network, float mutation_rate, float mutation_chance, MutationType mutation_type, double mutation_std_dev, int current_gen, int max_gens, double fitness_std_dev)\n\n"
        "Concept:\n"
        "Mutates a neural network's weights and biases in-place. "
        "Mutation is a genetic operator that introduces small, random changes into the network's parameters. "
        "This helps to maintain genetic diversity in the population and prevents the algorithm from getting stuck in local optima. "
        "This function acts as a dispatcher, calling the specific mutation method based on the `mutation_type` parameter."
    },
    {
        "select_fittest",
        "NetworkFitness* select_fittest(NetworkFitness* population_with_fitness, int population_size, int* num_fittest, SelectionType selection_type, int tournament_size)\n\n"
        "Concept:\n"
        "Selects a pool of fittest individuals from a population to act as parents. "
        "Selection is the process of choosing which individuals from the current generation will be parents for the next generation. "
        "Fitter individuals have a higher chance of being selected. "
        "This function acts as a dispatcher, calling the specific selection method (e.g., tournament, elitism) based on the `selection_type` parameter."
    },
    {NULL, NULL} // Sentinel value
};

const char* gann_get_doc(const char* function_name) {
    for (int i = 0; function_docs[i].function_name != NULL; i++) {
        if (strcmp(function_docs[i].function_name, function_name) == 0) {
            return function_docs[i].documentation;
        }
    }
    return NULL;
}
