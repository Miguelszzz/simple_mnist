#!/usr/bin/env bash
# Download and prepare the MNIST dataset for SIMPLE-MNIST
# This script downloads the MNIST dataset from Google Cloud Storage,
# which is a reliable mirror of the original dataset.

# Print colored status messages
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create a directory for the dataset
mkdir -p mnist_data
cd mnist_data

# Download MNIST files directly from Google Cloud Storage
echo -e "${BLUE}Downloading MNIST dataset from Google Cloud Storage...${NC}"
echo "This may take a few moments depending on your internet connection."

files=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

for file in "${files[@]}"; do
  echo -e "Downloading ${file}..."
  wget -q --show-progress https://storage.googleapis.com/cvdf-datasets/mnist/${file}

  # Check if download was successful
  if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to download ${file}${NC}"
    echo "Please check your internet connection and try again."
    exit 1
  fi
done

# Extract the files
echo -e "\n${BLUE}Extracting files...${NC}"
for file in "${files[@]}"; do
  echo "Extracting ${file}..."
  gunzip -f ${file}
done

# Verify files
missing_files=0
for file in "train-images-idx3-ubyte" "train-labels-idx1-ubyte" "t10k-images-idx3-ubyte" "t10k-labels-idx1-ubyte"; do
  if [ ! -f ${file} ]; then
    echo -e "${RED}Error: ${file} is missing${NC}"
    missing_files=$((missing_files + 1))
  fi
done

if [ ${missing_files} -gt 0 ]; then
  echo -e "${RED}Error: Failed to extract ${missing_files} MNIST files${NC}"
  exit 1
fi

# Display file sizes and information
echo -e "\n${BLUE}MNIST Dataset Information:${NC}"
echo -e "File sizes:"
ls -lh

# Calculate total size
total_size=$(du -ch *.ubyte | grep total | cut -f1)
echo -e "\nTotal dataset size: ${total_size}"

# Print summary statistics
train_images_size=$(stat -f%z "train-images-idx3-ubyte" 2>/dev/null || stat --format="%s" "train-images-idx3-ubyte")
train_images_count=$((${train_images_size} - 16))
train_images_count=$((${train_images_count} / 784))

test_images_size=$(stat -f%z "t10k-images-idx3-ubyte" 2>/dev/null || stat --format="%s" "t10k-images-idx3-ubyte")
test_images_count=$((${test_images_size} - 16))
test_images_count=$((${test_images_count} / 784))

echo -e "\nDataset contains:"
echo "- ${train_images_count} training images (28x28 pixels)"
echo "- ${test_images_count} test images (28x28 pixels)"

echo -e "\n${GREEN}MNIST dataset has been downloaded and prepared successfully${NC}"
echo -e "Path to MNIST data: $(pwd)"
echo -e "You can use this path as an argument to simple_mnist:"
echo -e "${BLUE}./simple_mnist $(pwd)${NC}"
echo -e "Or if you're in the build directory:"
echo -e "${BLUE}./simple_mnist ./mnist_data${NC}"

exit 0
