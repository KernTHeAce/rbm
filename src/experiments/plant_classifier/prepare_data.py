import os
import random
from pathlib import Path

from src import DATA_DIR

dataset_path = Path(DATA_DIR, "custom_plants_dataset_balanced")
classes = os.listdir(dataset_path)

min_class_number = 1610

for class_ in classes:
    images = os.listdir(dataset_path / class_)
    if (difference := len(images) - min_class_number) > 0:
        choices_for_deleting = random.choices(images, k=difference)
        for image in choices_for_deleting:
            if (path := dataset_path / class_ / image).exists():
                os.remove(path)
