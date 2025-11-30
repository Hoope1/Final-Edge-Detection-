
from abc import ABC, abstractmethod
import numpy as np
import torch
import cv2


class EdgeDetectorBase(ABC):
    """Gemeinsame Basisklasse für alle Kantendetektoren.

    Jede Unterklasse implementiert :meth:`load_model` und :meth:`detect` und
    nutzt dabei die hier vorbereiteten Convenience-Methoden für Pre- und
    Postprocessing.
    """

    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    @abstractmethod
    def load_model(self):
        """Lädt das vortrainierte Modell."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Führt die Kantenerkennung auf einem einzelnen Bild aus."""

    @staticmethod
    def preprocess(image: np.ndarray, target_size: tuple = None) -> np.ndarray:
        """Optionales Resize oder Normalisierung für Eingabebilder."""

        if target_size is not None:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return image

    @staticmethod
    def postprocess(edge_map: np.ndarray) -> np.ndarray:
        """Normiert, invertiert und clipt das Roh-Edge-Map auf uint8."""

        if edge_map.size == 0:
            return np.zeros_like(edge_map, dtype=np.uint8)

        if edge_map.dtype != np.uint8:
            max_val = float(np.max(edge_map))
            if max_val == 0:
                return np.zeros(edge_map.shape, dtype=np.uint8)
            edge_map = (255.0 * edge_map / max_val).clip(0, 255).astype(np.uint8)
        return 255 - edge_map  # invertiert: dunkle Kanten auf weißem Hintergrund
