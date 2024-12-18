# src/train.py
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
import config
import create_folds
import joblib
import model_dispatcher
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_model_for_multigpu(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    return model


def train_model_with_history(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

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
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Adjust learning rate
        scheduler.step(val_loss)  # Usar pérdida de validación

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Guardar el modelo final
    model_save_path = os.path.join(config.MODEL_OUTPUT, "custom_cnn.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Guardar en formato TorchScript
    # Crear un tensor de ejemplo
    example_input = torch.randn(1, 1, 224, 224).to(device)  # Asegúrate de que las dimensiones sean las correctas
    scripted_model = torch.jit.trace(model, example_input)

    scripted_model_path = os.path.join(config.MODEL_OUTPUT, "custom_cnn_scripted.pt")
    torch.jit.save(scripted_model, scripted_model_path)
    print(f"TorchScript model saved to {scripted_model_path}")

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }
    joblib.dump(metrics, os.path.join(config.MODEL_OUTPUT, "metrics.pkl"))

    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--num_epochs",
        type=int
    )
    args = parser.parse_args()
    model = model_dispatcher.models[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = prepare_model_for_multigpu(model)

    train_loader_cnn, val_loader_cnn = create_folds.return_dataset()
    train_labels = [label for _, label in train_loader_cnn.dataset]

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print("Training Custom CNN Model (from scratch)...")
    train_losses_cnn, val_losses_cnn, train_accuracies_cnn, val_accuracies_cnn = train_model_with_history(
        model, train_loader_cnn, val_loader_cnn, criterion, optimizer, args.num_epochs
    )
