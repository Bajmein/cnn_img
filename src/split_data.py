import os
import shutil
from glob import glob
import random

from typing_extensions import LiteralString


def redistribute_data(train_dir, val_dir, class_names, val_split=0.2, seed=42) -> None:
    random.seed(seed)
    for class_name in class_names:
        train_class_dir: LiteralString | str | bytes = os.path.join(train_dir, class_name)
        val_class_dir: LiteralString | str | bytes = os.path.join(val_dir, class_name)

        os.makedirs(val_class_dir, exist_ok=True)

        images: list[str | bytes] = glob(os.path.join(train_class_dir, "*"))
        random.shuffle(images)

        num_to_move: int = int(len(images) * val_split)

        print(f"Moviendo {num_to_move} imágenes de {class_name} a validación...")

        for img in images[:num_to_move]:
            shutil.move(img, val_class_dir)

    print("Redistribución completada.")


if __name__ == "__main__":
    train_dir: str = "C:/Users/bajme/PycharmProjects/clasificador_img/input/train"
    val_dir: str = "C:/Users/bajme/PycharmProjects/clasificador_img/input/val"
    class_names: list[str] = ["NORMAL", "PNEUMONIA"]

    redistribute_data(train_dir, val_dir, class_names, val_split=0.2)