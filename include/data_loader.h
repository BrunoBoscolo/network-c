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

/**
 * @brief Loads the MNIST dataset from the specified IDX-formatted files.
 * @param image_path The file path to the MNIST image data.
 * @param label_path The file path to the MNIST label data.
 * @return A pointer to the loaded Dataset, or NULL on failure.
 */
Dataset* load_mnist_dataset(const char* image_path, const char* label_path);

/**
 * @brief Creates a dummy dataset with random values for testing purposes.
 * @param num_items The number of items (images and labels) to create in the dataset.
 * @return A pointer to the created Dataset.
 */
Dataset* create_dummy_dataset(int num_items);

/**
 * @brief Frees the memory allocated for a dataset.
 * @param dataset The dataset to free.
 */
void free_dataset(Dataset* dataset);

#endif // DATA_LOADER_H
