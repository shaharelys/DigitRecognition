import os
import numpy as np
from typing import Tuple
from constants import IMG_SIZE


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset from the given path.

    Args:
    path (str): Path to the dataset.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Returns four numpy arrays, representing the training images,
    training labels, test images, and test labels respectively.
    """

    # Set paths
    train_images_path = os.path.join(path, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(path, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')

    # Set np arrays. Indexing is to skip file headers
    train_images = np.fromfile(train_images_path, dtype=np.uint8)[16:].reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    train_labels = np.fromfile(train_labels_path, dtype=np.uint8)[8:]
    test_images = np.fromfile(test_images_path, dtype=np.uint8)[16:].reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    test_labels = np.fromfile(test_labels_path, dtype=np.uint8)[8:]

    return train_images, train_labels, test_images, test_labels


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """
    Preprocesses the image data for the model. For the MNIST dataset, the images are already in grayscale and 28x28
    in size, so we'll just need to normalize the pixel values. Pixel values are integers that range from 0 (black) to
    255 (white). Normalizing these values will help our model train faster.

    Args:
    images (np.ndarray): Array of images to preprocess.

    Returns:
    np.ndarray: Array of preprocessed images.
    """

    # Normalize pixel values to [0, 1] range
    images = images.astype('float32') / 255

    return images
