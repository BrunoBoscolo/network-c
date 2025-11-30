# Neural Network Structure and .dat File Format Report

This report details the structure of the neural network, how its weights and biases are stored in matrices, and the binary format of the `.dat` files used for saving and loading the network.

## Neural Network and Matrix Structures

The neural network's architecture is defined by two primary structs: `NeuralNetwork` and `Matrix`.

### `Matrix` Struct

The `Matrix` struct is the fundamental building block for storing weights and biases. It is defined in `include/matrix.h` as follows:

```c
typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;
```

-   `rows`: The number of rows in the matrix.
-   `cols`: The number of columns in the matrix.
-   `data`: A dynamically allocated 2D array of `double` values that stores the matrix elements.

### `NeuralNetwork` Struct

The `NeuralNetwork` struct, defined in `include/neural_network.h`, represents the entire neural network and contains the following fields:

```c
typedef struct {
    int num_layers;
    int* architecture;
    Matrix** weights;
    Matrix** biases;
    ActivationType activation_hidden;
    ActivationType activation_output;
    OptimizerState* optimizer_state;
} NeuralNetwork;
```

-   `num_layers`: The total number of layers in the network, including the input, hidden, and output layers.
-   `architecture`: A dynamically allocated array of integers that specifies the number of neurons in each layer.
-   `weights`: An array of `Matrix` pointers. Each matrix `weights[i]` stores the weights connecting layer `i` to layer `i+1`.
-   `biases`: An array of `Matrix` pointers. Each matrix `biases[i]` stores the biases for the neurons in layer `i+1`.
-   `activation_hidden`: The activation function used for all hidden layers.
-   `activation_output`: The activation function used for the output layer.

## Storage of Weights and Biases

The weights and biases of the neural network are stored in arrays of `Matrix` pointers within the `NeuralNetwork` struct.

-   **Weights**: The `weights` array contains `num_layers - 1` matrices. Each matrix `weights[i]` has dimensions `architecture[i]` x `architecture[i+1]`, where `architecture[i]` is the number of neurons in the current layer and `architecture[i+1]` is the number of neurons in the next layer.

-   **Biases**: The `biases` array also contains `num_layers - 1` matrices. Each matrix `biases[i]` is a row vector with dimensions `1` x `architecture[i+1]`, containing the bias for each neuron in the next layer.

## .dat File Format

The `.dat` file is a binary file that stores a complete representation of the neural network, allowing it to be saved and loaded. The `nn_save` function in `lib/neural_network.c` writes the following data to the file in a specific order:

1.  **Header**:
    -   `num_layers` (int): The number of layers in the network.
    -   `activation_hidden` (ActivationType): The activation function for the hidden layers.
    -   `activation_output` (ActivationType): The activation function for the output layer.

2.  **Architecture**:
    -   `architecture` (array of ints): The number of neurons in each layer, with a total of `num_layers` integers.

3.  **Weights and Biases**:
    -   The weights and biases are written sequentially for each layer, starting from the first hidden layer. For each layer `i` from `0` to `num_layers - 2`:
        -   The `weights[i]` matrix is written row by row as a flat array of `double` values.
        -   The `biases[i]` matrix is written as a flat array of `double` values.

This binary format ensures that all the necessary information to reconstruct the neural network is stored in a compact and efficient manner.
