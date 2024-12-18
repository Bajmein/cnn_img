import os
import shutil
from glob import glob
import random


def redistribute_data(train_dir, val_dir, class_names, val_split=0.2, seed=42):
    random.seed(seed)
    for class_name in class_names:
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)

        # Crear directorio en validación si no existe
        os.makedirs(val_class_dir, exist_ok=True)

        # Listar imágenes en la carpeta de entrenamiento
        images = glob(os.path.join(train_class_dir, "*"))
        random.shuffle(images)

        # Calcular cuántas imágenes mover
        num_to_move = int(len(images) * val_split)

        print(f"Moviendo {num_to_move} imágenes de {class_name} a validación...")

        for img in images[:num_to_move]:
            shutil.move(img, val_class_dir)

    print("Redistribución completada.")


if __name__ == "__main__":
    train_dir = "C:/Users/bajme/PycharmProjects/clasificador_img/input/train"
    val_dir = "C:/Users/bajme/PycharmProjects/clasificador_img/input/val"
    class_names = ["NORMAL", "PNEUMONIA"]

    redistribute_data(train_dir, val_dir, class_names, val_split=0.2)
