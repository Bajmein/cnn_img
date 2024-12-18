import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Nueva capa convolucional reducida
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 16 * 16, 64)  # Reducción en el número de neuronas
        self.fc2 = nn.Linear(64, 32)  # Reducción adicional
        self.fc3 = nn.Linear(32, 2)   # Mantenido a 2 para la clasificación binaria
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


models = {
    "custom_cnn": CustomCNN(),
}
