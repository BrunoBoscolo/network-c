# Visualization of a Sample .dat File

This document provides a human-readable representation of a `.dat` file for a simple neural network. This visualization is intended for presentation purposes to clearly explain the structure and content of the binary file.

## Sample Neural Network Configuration

-   **Architecture**: 2 input neurons, 2 hidden neurons, 1 output neuron (`{2, 2, 1}`)
-   **Activation Functions**: `SIGMOID` for both hidden and output layers.

## .dat File Structure Visualization

A `.dat` file is a binary file, so it doesn't contain formatted text. However, if we were to visualize its structure and content in a human-readable format, it would look like this:

---

### 1. Header

| Data Type        | Field Name          | Example Value      | Description                                     |
| ---------------- | ------------------- | ------------------ | ----------------------------------------------- |
| `int`            | `num_layers`        | `3`                | The total number of layers (1 input + 1 hidden + 1 output). |
| `ActivationType` | `activation_hidden` | `0` (for SIGMOID)  | The activation function for the hidden layers.   |
| `ActivationType` | `activation_output` | `0` (for SIGMOID)  | The activation function for the output layer.   |

---

### 2. Architecture

| Data Type      | Field Name     | Example Values | Description                               |
| -------------- | -------------- | -------------- | ----------------------------------------- |
| `int[]`        | `architecture` | `{2, 2, 1}`    | An array of integers defining the number of neurons in each layer. |

---

### 3. Weights and Biases

#### Layer 1 (Input to Hidden)

-   **Weights Matrix (`weights[0]`)**: `2x2` matrix
-   **Biases Matrix (`biases[0]`)**: `1x2` matrix

| Data Type | Field Name     | Example Values                               |
| --------- | -------------- | -------------------------------------------- |
| `double`  | `weights[0]`   | `{0.1, 0.2, 0.3, 0.4}`                       |
| `double`  | `biases[0]`    | `{0.5, 0.6}`                                 |

#### Layer 2 (Hidden to Output)

-   **Weights Matrix (`weights[1]`)**: `2x1` matrix
-   **Biases Matrix (`biases[1]`)**: `1x1` matrix

| Data Type | Field Name     | Example Values |
| --------- | -------------- | -------------- |
| `double`  | `weights[1]`   | `{0.7, 0.8}`   |
| `double`  | `biases[1]`    | `{0.9}`        |

---

This visualization breaks down the binary `.dat` file into a clear and understandable format, making it easy to see how the neural network's configuration and parameters are stored.
