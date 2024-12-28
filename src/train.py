import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import joblib
import numpy as np
from colorama import Fore, Style
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config
import create_folds
import model_dispatcher
from losses import FocalLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = config.EPOCHS

def preparar_modelo_para_gpu(model):
    if torch.cuda.is_available():
        print(f"\nUsando GPU: {torch.cuda.get_device_name(0)}")
        model = model.to(device)
    else:
        print("GPU no disponible, usando CPU.")
        model = model.to('cpu')
    return model

def imprimir_resultados_epoch(epoch, train_loss, train_acc, val_loss, val_acc, precision, recall, f1, cm, auc_roc):
    print("=" * 60)
    print(f"Epoch: {Fore.LIGHTYELLOW_EX}{epoch}/{epochs}{Style.RESET_ALL}")
    print(f"AUC-ROC:         {Fore.CYAN}{auc_roc:.4f}{Style.RESET_ALL}")
    print()
    print(f"{Fore.BLUE}--- Métricas de Entrenamiento ---{Style.RESET_ALL}")
    print(f"Pérdida:         {train_loss:.4f}")
    print(f"Precisión:       {Fore.GREEN}{train_acc:.2f}%{Style.RESET_ALL}")
    print()
    print(f"{Fore.MAGENTA}--- Métricas de Validación ---{Style.RESET_ALL}")
    print(f"Pérdida:         {val_loss:.4f}")
    print(f"Precisión:       {Fore.GREEN}{val_acc:.2f}%{Style.RESET_ALL}")
    print(f"Precisión (Val): {precision:.4f}")
    print(f"Recall (Val):    {recall:.4f}")
    print(f"F1-Score (Val):  {f1:.4f}")
    print()
    print(f"{Fore.CYAN}--- Matriz de Confusión ---{Style.RESET_ALL}")
    print("          Predicho: Normal    Predicho: Neumonía")
    print(f"Normal     {cm[0, 0]:<15}{cm[0, 1]}")
    print(f"Neumonía   {cm[1, 0]:<15}{cm[1, 1]}")
    print("=" * 60)

def evaluar_modelo(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, cm

def entrenar_modelo_con_historial(model, train_loader, val_loader, criterion, optimizer, scheduler, accumulation_steps, num_epochs=config.EPOCHS):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            with autocast('cuda'):
                outputs = model(images, epoch=epoch)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

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

        all_val_labels = []
        all_val_predictions = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, epoch=epoch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_val_labels.append(labels.cpu().numpy())
                all_val_predictions.append(torch.softmax(outputs, dim=1).cpu().numpy()[:, 1])

        val_loss /= len(val_loader)
        val_loss_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_loss_accuracy)

        all_val_labels = np.concatenate(all_val_labels)
        all_val_predictions = np.concatenate(all_val_predictions)
        auc = roc_auc_score(all_val_labels, all_val_predictions)

        val_accuracy, val_precision, val_recall, val_f1, val_cm = evaluar_modelo(model, val_loader)

        imprimir_resultados_epoch(
            epoch + 1,
            train_loss,
            train_accuracy,
            val_loss,
            val_loss_accuracy,
            val_precision,
            val_recall,
            val_f1,
            val_cm,
            auc
        )

        scheduler.step(val_loss)

    torch.save(model.state_dict(), os.path.join(config.MODEL_OUTPUT, "custom_cnn.pth"))

    ejemplo_entrada = torch.randn(1, 1, config.TAMANO_IMAGEN, config.TAMANO_IMAGEN).to(device)
    modelo_torchscript = torch.jit.trace(model, ejemplo_entrada)
    torch.jit.save(modelo_torchscript, os.path.join(config.MODEL_OUTPUT, "custom_cnn_scripted.pt"))

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
        parser.add_argument("--model", type=str, help="Nombre del modelo a usar (de model_dispatcher)")
        args = parser.parse_args()

        model = model_dispatcher.models[args.model]()
        model = preparar_modelo_para_gpu(model)

        train_loader_cnn, val_loader_cnn, test_loader_cnn = create_folds.obtener_datasets()

        effective_batch_size = config.EFFECTIVE_BATCH_SIZE
        accumulation_steps = max(1, effective_batch_size // config.BATCH_SIZE)
        print(f"Pasos de acumulación: {Fore.CYAN}{accumulation_steps}{Style.RESET_ALL}\n")

        criterion = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            verbose=True
        )

        print("Entrenando el modelo CNN personalizado...\n")

        train_losses_cnn, val_losses_cnn, train_accuracies_cnn, val_accuracies_cnn = entrenar_modelo_con_historial(
            model, train_loader_cnn, val_loader_cnn, criterion, optimizer, scheduler, accumulation_steps, config.EPOCHS
        )

    except KeyboardInterrupt:
        print("Saliendo del entrenamiento antes de tiempo.")
