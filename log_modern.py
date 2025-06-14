
import logging
import os
from datetime import datetime


def setup_logger(log_dir="logs", log_level=logging.INFO):
    """
    Setzt ein globales Logging-Setup auf, das in Datei und Konsole schreibt.

    Args:
        log_dir (str): Zielverzeichnis für Log-Dateien
        log_level (int): Logging-Stufe (z.B. logging.DEBUG)
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"log_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")

    # Dateilog
    fh = logging.FileHandler(logfile)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Konsole
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Logger initialisiert.")
    logger.info(f"Logging in Datei: {logfile}")
