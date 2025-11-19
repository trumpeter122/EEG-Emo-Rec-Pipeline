import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D_N1(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 20, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(16, output_size)

    def forward(self, x):
        # Conv layers with ReLU, batch norm, and pooling
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers with activations and dropout
        x = torch.tanh(self.fc1(x))
        x = self.dropout1(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
