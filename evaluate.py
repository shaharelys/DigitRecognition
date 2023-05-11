from typing import Tuple
import numpy as np
from tensorflow.keras.models import Sequential


def evaluate_model(model: Sequential, test_images: np.ndarray, test_labels: np.ndarray) -> Tuple[float, float]:
    """
    Evaluates the model on the test data.

    Args:
    model (Sequential): The model to evaluate.
    test_images (np.ndarray): The test images.
    test_labels (np.ndarray): The test labels.

    Returns:
    Tuple[float, float]: The loss and accuracy of the model on the test data.
    """
    pass
