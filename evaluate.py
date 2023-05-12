# evaluate.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import load_dataset, preprocess_images
from model import Net
from config import DATA_PATH, MODEL_PATH
from constants import BATCH_SIZE

# Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model() -> None:
    _, _, test_images, test_labels = load_dataset(DATA_PATH)

    # Preprocess images
    test_images = preprocess_images(test_images)

    # Convert numpy arrays to PyTorch tensors
    test_images = torch.tensor(test_images)
    test_labels = torch.tensor(test_labels)

    # Create dataloaders
    test_data = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    # Load the model
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on test images: {100 * correct / total}%')


if __name__ == "__main__":
    evaluate_model()
