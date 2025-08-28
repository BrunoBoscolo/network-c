#!/bin/bash

# Create a data directory if it doesn't exist
mkdir -p data

# Base URL for the MNIST dataset
BASE_URL="http://yann.lecun.com/exdb/mnist"

# Files to download
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

# Download and unzip each file
for file in "${FILES[@]}"; do
    if [ ! -f "data/${file%.gz}" ]; then
        echo "Downloading ${file}..."
        wget -q -O "data/${file}" "${BASE_URL}/${file}"
        echo "Unzipping ${file}..."
        gunzip "data/${file}"
    else
        echo "${file%.gz} already exists. Skipping."
    fi
done

echo "MNIST dataset is ready in the 'data/' directory."
