
import logging
import os
from typing import Dict, List

import numpy as np
import psutil
import torch
import yaml
from tqdm import tqdm

from detectors_modern import (
    BDCNDetector,
    CASENetDetector,
    DexiNedDetector,
    DummyDetector,
    HEDDetector,
    RCFDetector,
    StructuredForestDetector,
)
from utils_image import ensure_grayscale, list_valid_images, load_image, resize_to_target, save_image


def load_config(config_path: str = "config_modern.yaml") -> Dict:
    """Lädt die YAML-Konfiguration."""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_ram_usage_percent() -> float:
    """Gibt die aktuelle RAM-Auslastung zurück."""

    return psutil.virtual_memory().percent


def get_detector(name: str, config: Dict):
    """Instanziiert den gewünschten Detektor basierend auf der Konfiguration."""

    model_paths = config["models"]
    use_cuda = config.get("use_cuda", True)

    if name == "structured_forest":
        return StructuredForestDetector(model_paths["structured_forest"], use_cuda=False)
    if name == "bdcn":
        return BDCNDetector(model_paths["bdcn"], use_cuda)
    if name == "rcf":
        return RCFDetector(model_paths["rcf"], use_cuda)
    if name == "hed":
        return HEDDetector(
            proto_path=model_paths["hed"]["proto"],
            model_path=model_paths["hed"]["model"],
            use_cuda=use_cuda,
        )
    if name == "dexined":
        return DexiNedDetector(model_paths["dexined"], use_cuda)
    if name == "casenet":
        return CASENetDetector(model_paths["casenet"], use_cuda)
    return DummyDetector()


def process_images(methods: List[str], config: Dict) -> Dict[str, List[str]]:
    """Verarbeitet alle gültigen Bilder mit den ausgewählten Methoden."""

    image_dir = config["image_dir"]
    result_dir = config["result_dir"]
    target_res = tuple(config.get("target_resolution", [2480, 3508]))
    resize_enabled = config.get("resize_output", True)
    preserve_aspect = config.get("preserve_aspect_ratio", True)
    invert_enabled = config.get("invert_output", True)

    image_paths = list_valid_images(image_dir)
    if not image_paths:
        logging.warning("Keine gültigen Eingabebilder in %s gefunden.", image_dir)
        return {method: [] for method in methods}

    os.makedirs(result_dir, exist_ok=True)
    results = {method: [] for method in methods}

    for method in methods:
        detector = get_detector(method, config)
        detector.load_model()
        output_dir = os.path.join(result_dir, method)
        os.makedirs(output_dir, exist_ok=True)

        for path in tqdm(image_paths, desc=f"Verarbeite mit {method.upper()}"):
            image = load_image(path)
            orig_shape = image.shape[:2]

            # Optionale Skalierung (ohne Verzerrung)
            if resize_enabled:
                h, w = orig_shape
                th, tw = target_res[1], target_res[0]
                if h < th or w < tw:
                    scale = min(tw / w, th / h) if preserve_aspect else 1
                    new_size = (int(w * scale), int(h * scale))
                    image = resize_to_target(image, new_size)

            edge_map = detector.detect(image)

            # Sicherstellen, dass Ausgabe dieselbe Größe wie das (ggf. veränderte) Eingabebild hat
            edge_map = ensure_grayscale(edge_map)
            if edge_map.shape != image.shape[:2]:
                edge_map = resize_to_target(edge_map, (image.shape[1], image.shape[0]))

            # Invertierung, falls aktiviert
            if invert_enabled:
                edge_map = 255 - edge_map

            filename = os.path.splitext(os.path.basename(path))[0] + "." + config["output_format"]
            save_path = os.path.join(output_dir, filename)
            save_image(save_path, edge_map)
            results[method].append(save_path)

            # RAM-Check
            ram_percent = get_ram_usage_percent()
            if ram_percent > config.get("max_ram_usage_percent", 90):
                logging.warning(f"[{method}] RAM-Auslastung überschritten: {ram_percent:.1f}%")
                break

    return results
