# src/train.py
import warnings

warnings.filterwarnings('ignore')  # Esto desactiva todas las advertencias

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_model_for_gpu(model):
    if torch.cuda.is_available():
        print()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        model = model.to(device)  # Mueve el modelo a la GPU
    else:
        print("GPU not available, using CPU instead.")
        model = model.to('cpu')  # Mueve el modelo a la CPU si no hay GPU
    return model


def apply_smote(train_loader):
    print()
    print("Applying SMOTE for data balancing...")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    data, labels = zip(*[(x.view(-1).numpy(), y.item()) for x, y in train_loader.dataset])
    data, labels = np.array(data), np.array(labels)

    # Generar datos sintéticos
    data_resampled, labels_resampled = smote.fit_resample(data, labels)
    data_resampled = data_resampled.reshape(-1, 1, 256, 256)  # Ajustar dimensiones
    return torch.utils.data.TensorDataset(
        torch.tensor(data_resampled, dtype=torch.float32),
        torch.tensor(labels_resampled, dtype=torch.long)
    )


def print_epoch_results(epoch, train_loss, train_acc, val_loss, val_acc, precision, recall, f1, cm):
    print(
        f"Epoch [{epoch}/30] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, "
        f"Val Acc: {val_acc:.2f}%")
    print(f"Validation Metrics: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    cm_line = "Confusion Matrix: " + " ".join(map(str, cm.flatten()))  # Aplana la matriz y la convierte en una sola línea
    print(cm_line)
    print()  # Espacio adicional entre épocas y matrices de confusión


def evaluate_model(model, dataloader, device='cuda'):
    model.eval()  # Establece el modelo en modo de evaluación
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No se necesita cálculo de gradientes durante la evaluación
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Mover tanto entradas como etiquetas al dispositivo

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # Obtiene las predicciones y las mueve a CPU
            all_labels.extend(labels.cpu().numpy())  # Obtiene las etiquetas reales y las mueve a CPU

    # Calcular las métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')  # O 'micro','macro' según clasificación
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, cm


# Acumulación de gradientes
accumulation_steps = 4  # Acumula gradientes durante 4 pasos


def train_model_with_history(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Inicializa el scaler para la precisión mixta
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad()  # Inicia el gradiente acumulado

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Habilita la precisión mixta
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Escala el loss y calcula los gradientes
            scaler.scale(loss).backward()

            # Solo actualiza los parámetros después de N pasos
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Reinicia los gradientes para el siguiente conjunto de pasos

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validación
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
        val_loss_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_loss_accuracy)

        # Evaluación de métricas
        val_accuracy, val_precision, val_recall, val_f1, val_cm = evaluate_model(model, val_loader)

        # Usamos la nueva función para imprimir los resultados
        print_epoch_results(epoch + 1, train_loss, train_accuracy, val_loss, val_loss_accuracy, val_precision,
                            val_recall, val_f1, val_cm)

        # Ajusta la tasa de aprendizaje
        scheduler.step(val_loss)

    # Guardar el modelo y las métricas
    model_save_path = os.path.join(config.MODEL_OUTPUT, "custom_cnn.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    example_input = torch.randn(1, 1, 256, 256).to(device)
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
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str)
        parser.add_argument("--num_epochs", type=int)
        args = parser.parse_args()

        model = model_dispatcher.models[args.model]
        model = model.to(device)
        model = prepare_model_for_gpu(model)

        train_loader_cnn, val_loader_cnn, _ = create_folds.return_dataset()

        # Apply SMOTE
        smote_dataset = apply_smote(train_loader_cnn)
        train_loader_cnn = torch.utils.data.DataLoader(smote_dataset, batch_size=32, shuffle=True)

        class_weights = torch.tensor([3.0, 1.0]).to(device)  # Pondera más la clase "normal"
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        print("Training Custom CNN Model with SMOTE...")
        print()
        train_losses_cnn, val_losses_cnn, train_accuracies_cnn, val_accuracies_cnn = train_model_with_history(
            model, train_loader_cnn, val_loader_cnn, criterion, optimizer, scheduler, args.num_epochs
        )
    except KeyboardInterrupt:
        print("Exiting from training early")
