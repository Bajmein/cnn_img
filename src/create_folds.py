import pandas as pd
import numpy as np
import os
import torch
from PIL import Image
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, TAMANO_IMAGEN, BATCH_SIZE


def crear_dataset(ruta_carpeta) -> pd.DataFrame:
    lista_imagenes: list = []
    for categoria in ['NORMAL', 'PNEUMONIA']:
        ruta_categoria: str = os.path.join(ruta_carpeta, categoria)
        for nombre_archivo in os.listdir(ruta_categoria):
            ruta_archivo = os.path.join(ruta_categoria, nombre_archivo)
            if os.path.isfile(ruta_archivo) and nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                lista_imagenes.append([ruta_archivo, categoria])
    return pd.DataFrame(lista_imagenes, columns=['ruta_archivo', 'etiqueta'])


def etiquetar_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Directorio de entrenamiento: {TRAIN_DIR}")

    try:
        df_entrenamiento: pd.DataFrame = crear_dataset(TRAIN_DIR)
        df_validacion: pd.DataFrame = crear_dataset(VAL_DIR)
        df_prueba: pd.DataFrame = crear_dataset(TEST_DIR)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Una o más carpetas están ausentes: {e}")

    if df_entrenamiento.empty or df_validacion.empty or df_prueba.empty:
        raise ValueError("Uno o más datasets están vacíos. Por favor, verifica la estructura de las carpetas y los datos.")

    mapeo_etiquetas: dict[str, int] = {'NORMAL': 0, 'PNEUMONIA': 1}
    df_entrenamiento['etiqueta'] = df_entrenamiento['etiqueta'].map(mapeo_etiquetas)
    df_validacion['etiqueta'] = df_validacion['etiqueta'].map(mapeo_etiquetas)
    df_prueba['etiqueta'] = df_prueba['etiqueta'].map(mapeo_etiquetas)

    for nombre_dataset, df in zip(['Entrenamiento', 'Validación', 'Prueba'], [df_entrenamiento, df_validacion, df_prueba]):
        if df['etiqueta'].isnull().any():
            print(f"Advertencia: Etiquetas no mapeadas encontradas en el dataset {nombre_dataset}.")
            raise ValueError(f"Etiquetas no mapeadas en el dataset {nombre_dataset}. Verifica los nombres de las carpetas o el mapeo de etiquetas.")

    return df_entrenamiento, df_validacion, df_prueba


class DatasetImagenes(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None) -> None:
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        ruta_img = self.dataframe.iloc[idx, 0]
        etiqueta = int(self.dataframe.iloc[idx, 1])
        img = Image.open(ruta_img).convert('L')

        if self.transform:
            img = self.transform(img)

        etiqueta = torch.tensor(etiqueta, dtype=torch.long)

        return img, etiqueta


def obtener_datasets() -> tuple:
    df_entrenamiento, df_validacion, df_prueba = etiquetar_dataset()

    transform_entrenamiento = transforms.Compose([
        transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    transform_val_prueba = transforms.Compose([
        transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset_entrenamiento = DatasetImagenes(df_entrenamiento, transform=transform_entrenamiento)
    dataset_validacion = DatasetImagenes(df_validacion, transform=transform_val_prueba)
    dataset_prueba = DatasetImagenes(df_prueba, transform=transform_val_prueba)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    data, etiquetas = zip(*[(x.view(-1).numpy(), y.item()) for x, y in dataset_entrenamiento])
    data, etiquetas = np.array(data), np.array(etiquetas)

    data_resampled, etiquetas_resampled = smote.fit_resample(data, etiquetas)
    data_resampled = data_resampled.reshape(-1, 1, TAMANO_IMAGEN, TAMANO_IMAGEN)

    dataset_smote = TensorDataset(torch.tensor(data_resampled, dtype=torch.float32),
                                   torch.tensor(etiquetas_resampled, dtype=torch.long))

    loader_entrenamiento = DataLoader(
        dataset_smote,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    loader_validacion = DataLoader(
        dataset_validacion,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    loader_prueba = DataLoader(dataset_prueba, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return loader_entrenamiento, loader_validacion, loader_prueba


if __name__ == "__main__":
    obtener_datasets()
