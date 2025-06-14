
from abc import ABC, abstractmethod
import numpy as np
import torch
import cv2


class EdgeDetectorBase(ABC):
    """
    Abstrakte Basisklasse für alle modernen Kantendetektoren.
    Definiert das Interface für die Anwendung eines Modells auf ein Bild.
    """

    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    @abstractmethod
    def load_model(self):
        """
        Lädt das vortrainierte Modell.
        Muss von jeder Unterklasse implementiert werden.
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Führt die Kantenerkennung auf einem einzelnen Bild aus.
        Muss von jeder Unterklasse implementiert werden.

        Args:
            image (np.ndarray): RGB-Bild (H, W, 3), Wertebereich [0, 255]

        Returns:
            np.ndarray: Binäres Kantenerkennungsbild (H, W), Wertebereich [0, 255]
        """
        pass

    @staticmethod
    def preprocess(image: np.ndarray, target_size: tuple = None) -> np.ndarray:
        """
        Optionales Preprocessing, z. B. Resize oder Normalisierung.

        Args:
            image (np.ndarray): RGB-Bild
            target_size (tuple): (width, height) falls Resize erwünscht

        Returns:
            np.ndarray: Vorverarbeitetes Bild
        """
        if target_size is not None:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return image

    @staticmethod
    def postprocess(edge_map: np.ndarray) -> np.ndarray:
        """
        Nachverarbeitung: Normalisierung, Invertierung, Clipping.

        Args:
            edge_map (np.ndarray): rohes Edge-Bild (float32 oder uint8)

        Returns:
            np.ndarray: normiertes & invertiertes Kantenergebnis (uint8)
        """
        if edge_map.dtype != np.uint8:
            edge_map = (255.0 * edge_map / np.max(edge_map)).clip(0, 255).astype(np.uint8)
        return 255 - edge_map  # invertiert: dunkle Kanten auf weißem Hintergrund
