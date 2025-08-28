#include "data_loader.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// Helper function to swap endianness (from big-endian to little-endian)
static int swap_endian(int val) {
    return ((val >> 24) & 0xff) |
           ((val << 8) & 0xff0000) |
           ((val >> 8) & 0xff00) |
           ((val << 24) & 0xff000000);
}

// Loads the MNIST dataset from the specified files
Dataset* load_mnist_dataset(const char* image_path, const char* label_path) {
    // --- Open Files ---
    FILE* image_file = fopen(image_path, "rb");
    FILE* label_file = fopen(label_path, "rb");
    if (!image_file || !label_file) {
        fprintf(stderr, "Error opening dataset files.\n");
        if (image_file) fclose(image_file);
        if (label_file) fclose(label_file);
        return NULL;
    }

    // --- Read Image File Header ---
    int magic, num_images, rows, cols;
    fread(&magic, sizeof(int), 1, image_file);
    magic = swap_endian(magic);
    fread(&num_images, sizeof(int), 1, image_file);
    num_images = swap_endian(num_images);
    fread(&rows, sizeof(int), 1, image_file);
    rows = swap_endian(rows);
    fread(&cols, sizeof(int), 1, image_file);
    cols = swap_endian(cols);

    if (magic != 2051) {
        fprintf(stderr, "Invalid image file magic number.\n");
        fclose(image_file);
        fclose(label_file);
        return NULL;
    }

    // --- Read Label File Header ---
    int label_magic, num_labels;
    fread(&label_magic, sizeof(int), 1, label_file);
    label_magic = swap_endian(label_magic);
    fread(&num_labels, sizeof(int), 1, label_file);
    num_labels = swap_endian(num_labels);

    if (label_magic != 2049) {
        fprintf(stderr, "Invalid label file magic number.\n");
        fclose(image_file);
        fclose(label_file);
        return NULL;
    }

    if (num_images != num_labels) {
        fprintf(stderr, "Number of images and labels do not match.\n");
        fclose(image_file);
        fclose(label_file);
        return NULL;
    }

    // --- Create Dataset Struct ---
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) {
        fclose(image_file);
        fclose(label_file);
        return NULL;
    }
    dataset->num_items = num_images;
    dataset->images = create_matrix(num_images, rows * cols);
    dataset->labels = create_matrix(num_images, MNIST_NUM_CLASSES);

    if (!dataset->images || !dataset->labels) {
        free_dataset(dataset); // free_dataset handles partial allocation
        fclose(image_file);
        fclose(label_file);
        return NULL;
    }

    // --- Read Data ---
    int image_size = rows * cols;
    unsigned char* image_buffer = (unsigned char*)malloc(image_size * sizeof(unsigned char));
    unsigned char label_buffer;

    for (int i = 0; i < num_images; i++) {
        // Read image
        fread(image_buffer, sizeof(unsigned char), image_size, image_file);
        for (int j = 0; j < image_size; j++) {
            dataset->images->data[i][j] = (double)image_buffer[j] / 255.0;
        }

        // Read label and one-hot encode
        fread(&label_buffer, sizeof(unsigned char), 1, label_file);
        for(int k=0; k < MNIST_NUM_CLASSES; k++) {
            dataset->labels->data[i][k] = 0.0;
        }
        dataset->labels->data[i][label_buffer] = 1.0;
    }

    // --- Cleanup ---
    free(image_buffer);
    fclose(image_file);
    fclose(label_file);

    printf("Successfully loaded %d items from the MNIST dataset.\n", num_images);

    return dataset;
}

// Creates a dummy dataset with random values
Dataset* create_dummy_dataset(int num_items) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) return NULL;

    dataset->num_items = num_items;
    dataset->images = create_matrix(num_items, MNIST_IMAGE_SIZE);
    dataset->labels = create_matrix(num_items, MNIST_NUM_CLASSES);

    if (!dataset->images || !dataset->labels) {
        free_matrix(dataset->images);
        free_matrix(dataset->labels);
        free(dataset);
        return NULL;
    }

    // Seed random number generator if not already seeded
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }

    // Fill images with random pixel values (0.0 to 1.0)
    for (int i = 0; i < num_items; i++) {
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            dataset->images->data[i][j] = (double)rand() / RAND_MAX;
        }
    }

    // Fill labels with random one-hot encoded vectors
    for (int i = 0; i < num_items; i++) {
        int random_class = rand() % MNIST_NUM_CLASSES;
        dataset->labels->data[i][random_class] = 1.0;
    }

    return dataset;
}

// Frees the memory allocated for a dataset
void free_dataset(Dataset* dataset) {
    if (!dataset) return;
    free_matrix(dataset->images);
    free_matrix(dataset->labels);
    free(dataset);
}
