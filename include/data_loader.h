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
 * @return A pointer to the loaded Dataset, or NULL on failure. The caller is
 *         responsible for freeing this dataset using `free_dataset()`.
 */
Dataset* load_mnist_dataset(const char* image_path, const char* label_path);

/**
 * @brief Creates a dummy dataset with random values for testing purposes.
 * @param num_items The number of items (images and labels) to create in the dataset.
 * @return A pointer to the created Dataset. The caller is responsible for
 *         freeing this dataset using `free_dataset()`.
 */
Dataset* create_dummy_dataset(int num_items);

/**
 * @brief Creates a dummy dataset with a specific label for all items, for testing purposes.
 * @param num_items The number of items to create.
 * @param label The integer label (0-9) to assign to all items.
 * @return A pointer to the created Dataset. The caller is responsible for
 *         freeing this dataset using `free_dataset()`.
 */
Dataset* create_dummy_dataset_with_label(int num_items, int label);

/**
 * @brief Splits a dataset into two new datasets by copying the data.
 *
 * This function creates two new datasets and copies the corresponding data
 * from the original dataset. The caller is responsible for freeing the two
 * new datasets (`out_dataset_1`, `out_dataset_2`) as well as the original
 * dataset.
 *
 * @param original The dataset to split.
 * @param split_size The number of items to put in the second dataset (`out_dataset_2`).
 * @param out_dataset_1 Pointer to the first output dataset struct.
 * @param out_dataset_2 Pointer to the second output dataset struct.
 */
void split_dataset(const Dataset* original, int split_size, Dataset* out_dataset_1, Dataset* out_dataset_2);

/**
 * @brief Frees the memory allocated for a dataset.
 * @param dataset The dataset to free.
 */
void free_dataset(Dataset* dataset);

#endif // DATA_LOADER_H
