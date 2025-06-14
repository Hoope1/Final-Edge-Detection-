
import os
import cv2
import numpy as np
import torch
from detector_base import EdgeDetectorBase
import logging

logger = logging.getLogger(__name__)


class DummyDetector(EdgeDetectorBase):
    def load_model(self):
        logger.info("DummyDetector verwendet keine echten Modelle.")

    def detect(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)


class StructuredForestDetector(EdgeDetectorBase):
    def __init__(self, model_path, use_cuda=True):
        super().__init__(use_cuda=False)
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelldatei nicht gefunden: {self.model_path}")
        self.model = cv2.ximgproc.createStructuredEdgeDetection(self.model_path)

    def detect(self, image: np.ndarray) -> np.ndarray:
        rgb_float = image.astype(np.float32) / 255.0
        edges = self.model.detectEdges(rgb_float)
        return self.postprocess(edges)


class BDCNDetector(EdgeDetectorBase):
    def __init__(self, model_path, use_cuda=True):
        super().__init__(use_cuda)
        self.model_path = model_path
        self.model = None

    def load_model(self):
        from bdcn_repo.models import bdcn
        self.model = bdcn.BDCN()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray) -> np.ndarray:
        import kornia
        with torch.no_grad():
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = kornia.color.rgb_to_grayscale(tensor)
            tensor = tensor.to(self.device)
            out = self.model(tensor)[-1]
            edge = out.squeeze().cpu().numpy()
            return self.postprocess(edge)


class RCFDetector(EdgeDetectorBase):
    def __init__(self, model_path, use_cuda=True):
        super().__init__(use_cuda)
        self.model_path = model_path
        self.model = None

    def load_model(self):
        from rcf_repo.models.rcf import RCF
        self.model = RCF()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray) -> np.ndarray:
        import kornia
        with torch.no_grad():
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = kornia.color.rgb_to_grayscale(tensor)
            tensor = tensor.to(self.device)
            out = self.model(tensor)
            edge = out.squeeze().cpu().numpy()
            return self.postprocess(edge)


class HEDDetector(EdgeDetectorBase):
    def __init__(self, proto_path, model_path, use_cuda=True):
        super().__init__(use_cuda)
        self.proto_path = proto_path
        self.model_path = model_path
        self.net = None

    def load_model(self):
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        if self.use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(self, image: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(image.shape[1], image.shape[0]),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
        self.net.setInput(blob)
        edge = self.net.forward()
        edge = edge[0, 0]
        edge = cv2.resize(edge, (image.shape[1], image.shape[0]))
        return self.postprocess(edge)


class DexiNedDetector(EdgeDetectorBase):
    def __init__(self, model_path, use_cuda=True):
        super().__init__(use_cuda)
        self.model_path = model_path
        self.model = None

    def load_model(self):
        from dexined_repo.model import DexiNed
        self.model = DexiNed()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray) -> np.ndarray:
        import kornia
        with torch.no_grad():
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = kornia.color.rgb_to_grayscale(tensor)
            tensor = tensor.to(self.device)
            out = self.model(tensor)
            edge = out.squeeze().cpu().numpy()
            return self.postprocess(edge)

class CASENetDetector(EdgeDetectorBase):
    def __init__(self, model_path, use_cuda=True):
        super().__init__(use_cuda)
        self.model_path = model_path
        self.model = None

    def load_model(self):
        from casenet_repo.model.casenet import CASENet
        self.model = CASENet(num_classes=1)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: np.ndarray) -> np.ndarray:
        import kornia
        with torch.no_grad():
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = kornia.color.rgb_to_grayscale(tensor)
            tensor = tensor.to(self.device)
            out = self.model(tensor)
            edge = out.squeeze().cpu().numpy()
            return self.postprocess(edge)