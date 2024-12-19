import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Capa convolucional 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # Capa convolucional 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # Capa convolucional 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # Capa convolucional 4
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)  # Capa final convolucional
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Cálculo dinámico del tamaño de entrada para fc1
        dummy_input = torch.zeros(1, 1, 256, 256)
        dummy_output = self._get_conv_output(dummy_input)
        self.fc1 = nn.Linear(dummy_output, 64)  # Ajustar con tamaño correcto
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 2)

        self.dropout = nn.Dropout(0.6)

    def _get_conv_output(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        return int(np.prod(x.size()))  # Calcula el tamaño de la salida

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


models = {
    "custom_cnn": CustomCNN(),
}
