
import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch

from detector_base import EdgeDetectorBase

logger = logging.getLogger(__name__)


class DummyDetector(EdgeDetectorBase):
    """Detektor für Tests – gibt ein komplett leeres Edge-Map zurück."""

    def load_model(self):
        logger.info("DummyDetector verwendet keine echten Modelle.")

    def detect(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)


class StructuredForestDetector(EdgeDetectorBase):
    """Wrapper für OpenCV Structured Forest mit CPU-Only Fallback."""

    def __init__(self, model_path: str, use_cuda: bool = True):
        super().__init__(use_cuda=False)
        self.model_path = model_path
        self.model: Optional[object] = None
        self._use_fallback = False

    def load_model(self):
        """Lädt das Structured-Forest-Modell oder aktiviert den Canny-Fallback."""

        if not os.path.exists(self.model_path):
            logger.warning("Modelldatei nicht gefunden – nutze Canny-Fallback.")
            self._use_fallback = True
            return

        try:
            ximgproc = getattr(cv2, "ximgproc", None)
            if ximgproc is None:
                raise RuntimeError("OpenCV ximgproc Modul nicht verfügbar")
            self.model = ximgproc.createStructuredEdgeDetection(self.model_path)
            logger.info("Structured Forest Modell geladen.")
        except Exception as exc:  # pragma: no cover - sicherer Fallback
            logger.warning("Structured Forest konnte nicht geladen werden: %s", exc)
            self._use_fallback = True

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Führt die Kantendetektion aus und fällt bei Bedarf auf Canny zurück."""

        if self.model is None or self._use_fallback:
            logger.debug("Nutze Canny-Fallback für Structured Forest.")
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            edges = cv2.Canny(gray, 100, 200)
            return self.postprocess(edges.astype(np.float32))

        rgb_float = image.astype(np.float32) / 255.0
        edges = self.model.detectEdges(rgb_float)
        return self.postprocess(edges)


class BDCNDetector(EdgeDetectorBase):
    """BDC-Net Implementierung für Kantenerkennung."""

    def __init__(self, model_path: str, use_cuda: bool = True):
        super().__init__(use_cuda)
        self.model_path = model_path
        self.model: Optional[torch.nn.Module] = None

    def load_model(self):
        """Lädt das vortrainierte BDCN-Modell."""

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"BDCNet Checkpoint fehlt: {self.model_path}")

        from bdcn_repo.models import bdcn

        self.model = bdcn.BDCN()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Berechnet ein Edge-Map mit BDCN."""

        import kornia

        with torch.no_grad():
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = kornia.color.rgb_to_grayscale(tensor)
            tensor = tensor.to(self.device)
            out = self.model(tensor)[-1]
            edge = out.squeeze().cpu().numpy()
            return self.postprocess(edge)


class RCFDetector(EdgeDetectorBase):
    """RCF-Detektor (Richer Convolutional Features)."""

    def __init__(self, model_path: str, use_cuda: bool = True):
        super().__init__(use_cuda)
        self.model_path = model_path
        self.model: Optional[torch.nn.Module] = None

    def load_model(self):
        """Lädt das RCF-Modell."""

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RCF Checkpoint fehlt: {self.model_path}")

        from rcf_repo.models.rcf import RCF

        self.model = RCF()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Berechnet ein Edge-Map mit RCF."""

        import kornia

        with torch.no_grad():
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = kornia.color.rgb_to_grayscale(tensor)
            tensor = tensor.to(self.device)
            out = self.model(tensor)
            edge = out.squeeze().cpu().numpy()
            return self.postprocess(edge)


class HEDDetector(EdgeDetectorBase):
    """Holistically-Nested Edge Detection (OpenCV DNN)."""

    def __init__(self, proto_path: str, model_path: str, use_cuda: bool = True):
        super().__init__(use_cuda)
        self.proto_path = proto_path
        self.model_path = model_path
        self.net: Optional[cv2.dnn_Net] = None

    def load_model(self):
        """Lädt das HED-Caffe Modell."""

        if not os.path.exists(self.proto_path) or not os.path.exists(self.model_path):
            raise FileNotFoundError("HED Prototxt oder Modell fehlt.")

        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        if self.use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Berechnet ein Edge-Map mit HED."""

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(image.shape[1], image.shape[0]),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        edge = self.net.forward()[0, 0]
        edge = cv2.resize(edge, (image.shape[1], image.shape[0]))
        return self.postprocess(edge)


class DexiNedDetector(EdgeDetectorBase):
    """DexiNed-Detektor."""

    def __init__(self, model_path: str, use_cuda: bool = True):
        super().__init__(use_cuda)
        self.model_path = model_path
        self.model: Optional[torch.nn.Module] = None

    def load_model(self):
        """Lädt das DexiNed-Modell."""

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"DexiNed Checkpoint fehlt: {self.model_path}")

        from dexined_repo.model import DexiNed

        self.model = DexiNed()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Berechnet ein Edge-Map mit DexiNed."""

        import kornia

        with torch.no_grad():
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = kornia.color.rgb_to_grayscale(tensor)
            tensor = tensor.to(self.device)
            out = self.model(tensor)
            edge = out.squeeze().cpu().numpy()
            return self.postprocess(edge)

class CASENetDetector(EdgeDetectorBase):
    """CASENet-Detektor."""

    def __init__(self, model_path: str, use_cuda: bool = True):
        super().__init__(use_cuda)
        self.model_path = model_path
        self.model: Optional[torch.nn.Module] = None

    def load_model(self):
        """Lädt das CASENet-Modell."""

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"CASENet Checkpoint fehlt: {self.model_path}")

        from casenet_repo.model.casenet import CASENet

        self.model = CASENet(num_classes=1)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Berechnet ein Edge-Map mit CASENet."""

        import kornia

        with torch.no_grad():
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = kornia.color.rgb_to_grayscale(tensor)
            tensor = tensor.to(self.device)
            out = self.model(tensor)
            edge = out.squeeze().cpu().numpy()
            return self.postprocess(edge)
