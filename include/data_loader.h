#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "neural_network.h"

#define MNIST_IMAGE_ROWS 28
#define MNIST_IMAGE_COLS 28
#define MNIST_IMAGE_SIZE (MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS)
#define MNIST_NUM_CLASSES 10

// Represents a dataset of images and labels
typedef struct {
    int num_items;
    Matrix* images; // Each row is a flattened image
    Matrix* labels; // Each row is a one-hot encoded label
} Dataset;

// --- Data Loader Functions ---

// --- Data Loader Functions ---

// Loads the MNIST dataset from the specified files
Dataset* load_mnist_dataset(const char* image_path, const char* label_path);

// Creates a dummy dataset with random values
Dataset* create_dummy_dataset(int num_items);

// Frees the memory allocated for a dataset
void free_dataset(Dataset* dataset);

#endif // DATA_LOADER_H
