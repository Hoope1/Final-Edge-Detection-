
import os
import yaml
import time
import logging
import psutil
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from detectors_modern import (
    DummyDetector, StructuredForestDetector,
    BDCNDetector, RCFDetector, HEDDetector
)
from utils_image import (
    list_valid_images, load_image, save_image,
    resize_to_target, ensure_grayscale
)


def load_config(config_path: str = "config_modern.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_ram_usage_percent() -> float:
    return psutil.virtual_memory().percent


def get_detector(name: str, config: Dict):
    model_paths = config["models"]
    use_cuda = config.get("use_cuda", True)

    if name == "structured_forest":
        return StructuredForestDetector(model_paths["structured_forest"], use_cuda=False)
    elif name == "bdcn":
        return BDCNDetector(model_paths["bdcn"], use_cuda)
    elif name == "rcf":
        return RCFDetector(model_paths["rcf"], use_cuda)
    elif name == "hed":
        return HEDDetector(
            proto_path=model_paths["hed"]["proto"],
            model_path=model_paths["hed"]["model"],
            use_cuda=use_cuda
        )
    else:
        return DummyDetector()


def process_images(methods: List[str], config: Dict) -> Dict[str, List[str]]:
    image_dir = config["image_dir"]
    result_dir = config["result_dir"]
    target_res = tuple(config.get("target_resolution", [2480, 3508]))
    resize_enabled = config.get("resize_output", True)
    preserve_aspect = config.get("preserve_aspect_ratio", True)
    invert_enabled = config.get("invert_output", True)

    image_paths = list_valid_images(image_dir)
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
