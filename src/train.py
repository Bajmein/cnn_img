# src/train.py
import warnings

from torch.utils.data import TensorDataset
from traitlets.config.loader import ArgumentParser

from losses import FocalLoss

warnings.filterwarnings('ignore')

import argparse
import torch
import torch.optim as optim
import os
import config
import create_folds
import joblib
import model_dispatcher
from torch.optim.lr_scheduler import ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.amp import GradScaler, autocast
import torch.nn as nn


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs: int = 15


def prepare_model_for_gpu(model) -> any:
    if torch.cuda.is_available():
        print()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        model = model.to(device)
    else:
        print("GPU not available, using CPU instead.")
        model = model.to('cpu')
    return model


def apply_smote(train_loader) -> TensorDataset:
    print()
    print("Applying SMOTE for data balancing...")
    smote: SMOTE = SMOTE(sampling_strategy='auto', random_state=42)
    data, labels = zip(*[(x.view(-1).numpy(), y.item()) for x, y in train_loader.dataset])
    data, labels = np.array(data), np.array(labels)

    data_resampled, labels_resampled = smote.fit_resample(data, labels)
    data_resampled = data_resampled.reshape(-1, 1, 256, 256)
    return torch.utils.data.TensorDataset(
        torch.tensor(data_resampled, dtype=torch.float32),
        torch.tensor(labels_resampled, dtype=torch.long)
    )


def print_epoch_results(epoch, train_loss, train_acc, val_loss, val_acc, precision, recall, f1, cm) -> None:
    print(
        f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, "
        f"Val Acc: {val_acc:.2f}%")
    print(f"Validation Metrics: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    cm_line: str = "Confusion Matrix: " + " ".join(map(str, cm.flatten()))
    print(cm_line)
    print()


def evaluate_model(model, dataloader, device='cuda') -> tuple:
    model.eval()
    all_preds: list = []
    all_labels: list = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy: float | int = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, cm


accumulation_steps: int = 4


def train_model_with_history(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25) -> tuple:
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []

    scaler: GradScaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss: float = running_loss / len(train_loader)
        train_accuracy: float = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss: float = 0.0
        val_correct: int = 0
        val_total: int = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_loss_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_loss_accuracy)

        val_accuracy, val_precision, val_recall, val_f1, val_cm = evaluate_model(model, val_loader)

        print_epoch_results(epoch + 1, train_loss, train_accuracy, val_loss, val_loss_accuracy, val_precision,
                            val_recall, val_f1, val_cm)

        scheduler.step(val_loss)

    model_save_path: str = os.path.join(config.MODEL_OUTPUT, "custom_cnn.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    example_input: torch.Tensor = torch.randn(1, 1, 256, 256).to(device)
    scripted_model = torch.jit.trace(model, example_input)

    scripted_model_path: str = os.path.join(config.MODEL_OUTPUT, "custom_cnn_scripted.pt")
    torch.jit.save(scripted_model, scripted_model_path)
    print(f"TorchScript model saved to {scripted_model_path}")

    metrics: dict[str, list[float]] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }
    joblib.dump(metrics, os.path.join(config.MODEL_OUTPUT, "metrics.pkl"))

    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == "__main__":
    try:
        parser: ArgumentParser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str)
        parser.add_argument("--num_epochs", type=int)
        args = parser.parse_args()

        model = model_dispatcher.models[args.model]
        model = model.to(device)
        model = prepare_model_for_gpu(model)

        train_loader_cnn, val_loader_cnn, _ = create_folds.return_dataset()

        smote_dataset: TensorDataset = apply_smote(train_loader_cnn)
        train_loader_cnn: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]] = torch.utils.data.DataLoader(
            smote_dataset,
            batch_size=32,
            shuffle=True
        )

        # class_weights: torch.Tensor = torch.tensor([4.0, 1.0]).to(device)
        # criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(weight=class_weights)
        criterion: FocalLoss = FocalLoss(alpha=1, gamma=3)

        optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=0.0007, weight_decay=1e-4)
        scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        print("Training Custom CNN Model with SMOTE...")
        print()
        train_losses_cnn, val_losses_cnn, train_accuracies_cnn, val_accuracies_cnn = train_model_with_history(
            model, train_loader_cnn, val_loader_cnn, criterion, optimizer, scheduler, args.num_epochs
        )
    except KeyboardInterrupt:
        print("Exiting from training early")
