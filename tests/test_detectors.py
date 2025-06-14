
import os
import numpy as np
from detectors_modern import StructuredForestDetector, DummyDetector
from utils_image import save_image

def generate_dummy_image(width=512, height=512):
    """
    Erzeugt ein Dummy-RGB-Testbild (Gradient + Noise)
    """
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    image = np.stack([xv, yv, 255 - xv], axis=-1)
    noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
    return np.clip(image + noise, 0, 255)

def run_test_structured():
    """
    Testet den StructuredForestDetector mit Dummybild.
    """
    model_path = "models/structured/model.yml.gz"
    assert os.path.exists(model_path), "Modell fehlt"
    detector = StructuredForestDetector(model_path, use_cuda=False)
    detector.load_model()

    test_img = generate_dummy_image()
    edge_map = detector.detect(test_img)
    assert edge_map.shape[:2] == test_img.shape[:2]
    assert edge_map.dtype == np.uint8
    save_image("test_structured_result.png", edge_map)
    print("✅ StructuredForestDetector erfolgreich getestet")

def run_test_dummy():
    """
    Testet DummyDetector.
    """
    detector = DummyDetector()
    test_img = generate_dummy_image()
    edge_map = detector.detect(test_img)
    assert np.all(edge_map == 0)
    print("✅ DummyDetector erfolgreich getestet")

if __name__ == "__main__":
    run_test_dummy()
    run_test_structured()
