# model.py

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import load_dataset, preprocess_images
from sklearn.model_selection import train_test_split
from config import DATA_PATH, MODEL_PATH
from constants import BATCH_SIZE, EPOCHS, LEARNING_RATE

# Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'torch is using {device} as device')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


def train_model() -> None:
    train_images, train_labels, test_images, test_labels = load_dataset(DATA_PATH)

    # Preprocess images
    train_images, test_images = preprocess_images(train_images), preprocess_images(test_images)

    # Convert numpy arrays to PyTorch tensors
    train_images = torch.tensor(train_images)
    test_images = torch.tensor(test_images)
    train_labels = torch.tensor(train_labels).long()
    test_labels = torch.tensor(test_labels).long()

    # Create dataloaders
    train_data = TensorDataset(train_images, train_labels)
    test_data = TensorDataset(test_images, test_labels)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    # Create the model
    model = Net().to(device)

    # Define the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    model.train()

    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_loader):
            print(
                f'Running batch #{i}, epoch #{epoch + 1} | {("{:.2f}".format(float(i * BATCH_SIZE)*100 / 60000))}%')
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    train_model()
