import os
import shutil
import random
from glob import glob
from typing_extensions import LiteralString
from config import TRAIN_DIR, VAL_DIR, CLASES


def redistribute_data(train_dir, val_dir, class_names, val_split=0.1, seed=42) -> None:
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
    train_dir: str = TRAIN_DIR
    val_dir: str = VAL_DIR
    class_names: list[str] = CLASES

    redistribute_data(train_dir, val_dir, class_names, val_split=0.2)