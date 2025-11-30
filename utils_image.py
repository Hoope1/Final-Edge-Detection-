
import logging
import os
from typing import List, Tuple

import cv2
import numpy as np


SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.jp2'}


def list_valid_images(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        logging.warning("Eingabeverzeichnis %s existiert nicht.", input_dir)
        return []

    valid_images = []
    for file in os.listdir(input_dir):
        ext = os.path.splitext(file)[-1].lower()
        if ext in SUPPORTED_FORMATS:
            valid_images.append(os.path.join(input_dir, file))
    return sorted(valid_images)


def load_image(path: str) -> np.ndarray:
    """Lädt ein Bild als RGB-Array."""

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(path: str, image: np.ndarray):
    """Speichert ein RGB- oder Graustufenbild."""

    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def resize_to_target(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image
