import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import TAMANO_IMAGEN, DROPOUT_BASE_RATE, DROPOUT_DECAY, DROPOUT_MIN_RATE



class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        dummy_input = torch.zeros(1, 1, TAMANO_IMAGEN, TAMANO_IMAGEN)
        dummy_output = self._get_conv_output(dummy_input)

        self.fc1 = nn.Linear(dummy_output, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)

        self.dropout = AdaptiveDropout(
            base_rate=DROPOUT_BASE_RATE,
            decay=DROPOUT_DECAY,
            min_rate=DROPOUT_MIN_RATE
        )

    def _get_conv_output(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        return int(np.prod(x.size()))

    def forward(self, x, epoch=0):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x, epoch)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class AdaptiveDropout(nn.Module):
    def __init__(self, base_rate=0.5, decay=0.0, min_rate=0.0):
        super(AdaptiveDropout, self).__init__()
        self.base_rate = base_rate
        self.decay = decay
        self.min_rate = min_rate
        self.dynamic = decay > 0.0

    def forward(self, x, epoch=None):
        rate = self.base_rate
        if self.dynamic and epoch is not None:
            rate = max(self.min_rate, self.base_rate - epoch * self.decay)
        return F.dropout(x, p=rate, training=self.training)


models: dict[str, callable] = {
    "custom_cnn": lambda: CustomCNN(),
}