from imblearn.over_sampling import SMOTE
import numpy as np
import torch
from torch.utils.data import TensorDataset


def apply_smote(features, labels) -> tuple:
    smote: SMOTE = SMOTE()
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    return features_resampled, labels_resampled


def extract_latent_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.extend(targets.cpu().numpy())
    return np.vstack(features), np.array(labels)


def generate_synthetic_data(model, data_loader, device, batch_size) -> torch.utils.data.DataLoader:
    features, labels = extract_latent_features(model, data_loader, device)
    features_smote, labels_smote = apply_smote(features, labels)

    tensor_x: torch.Tensor = torch.Tensor(features_smote)
    tensor_y: torch.LongTensor = torch.LongTensor(labels_smote)
    dataset: TensorDataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
