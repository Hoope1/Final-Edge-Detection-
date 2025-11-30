"""Dienstprogramm zum automatischen Download aller benötigten Modelle."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional

import requests

logger = logging.getLogger(__name__)


class ModelArtifact(NamedTuple):
    """Beschreibt eine herunterladbare Modelldatei."""

    url: str
    path: Path
    description: str
    is_gdrive: bool = False


class DownloadError(RuntimeError):
    """Fehler beim Herunterladen eines Modells."""


GDRIVE_TEMPLATE = "https://drive.google.com/uc?export=download&id={file_id}"


def _ensure_parent_directory(path: Path) -> None:
    """Erstellt das Elternverzeichnis einer Datei."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_gdrive_token(response: requests.Response) -> Optional[str]:
    """Ermittelt ggf. ein Bestätigungstoken aus Google-Drive-Cookies."""

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _stream_to_file(response: requests.Response, destination: Path) -> None:
    """Schreibt einen Stream sicher auf die Festplatte."""

    chunk_size = 1024 * 1024
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                handle.write(chunk)


def download_file(url: str, destination: Path, is_gdrive: bool = False) -> None:
    """Lädt eine Datei per HTTP/HTTPS herunter.

    Args:
        url: Ziel-URL oder Google-Drive-File-ID.
        destination: Pfad zur Zieldatei.
        is_gdrive: Flag, ob es sich um einen Google-Drive-Download handelt.
    """

    _ensure_parent_directory(destination)
    session = requests.Session()

    try:
        if is_gdrive:
            initial = session.get(GDRIVE_TEMPLATE.format(file_id=url), stream=True, timeout=30)
            token = _extract_gdrive_token(initial)
            if token:
                download = session.get(
                    GDRIVE_TEMPLATE.format(file_id=url), params={"confirm": token}, stream=True, timeout=30
                )
            else:
                download = initial
        else:
            download = session.get(url, stream=True, timeout=30)

        download.raise_for_status()
        _stream_to_file(download, destination)
        logger.info("✅ %s erfolgreich heruntergeladen", destination)
    except Exception as exc:  # pragma: no cover - defensiv gegen Netzwerkfehler
        raise DownloadError(f"Download fehlgeschlagen für {destination}: {exc}") from exc


def build_artifacts_from_config(config: Dict) -> List[ModelArtifact]:
    """Erzeugt Artefakt-Liste aus der YAML-Konfiguration."""

    models = config.get("models", {})
    return [
        ModelArtifact(
            url="https://github.com/zijundeng/BDCN/releases/download/v1.0.0/bdcn_pretrained.pth",
            path=Path(models.get("bdcn", "bdcn_repo/pretrained/bdcn_pretrained.pth")),
            description="BDCNet Gewicht",
        ),
        ModelArtifact(
            url="1qxW3Z4Y6z3dpZJkZHZbwAb29rT1U3pS8",
            path=Path(models.get("rcf", "rcf_repo/model/RCF.pth")),
            description="RCF Gewicht",
            is_gdrive=True,
        ),
        ModelArtifact(
            url="https://github.com/s9xie/hed/raw/master/examples/hed/hed_pretrained_bsds.caffemodel",
            path=Path(models.get("hed", {}).get("model", "hed_repo/hed_pretrained_bsds.caffemodel")),
            description="HED Caffe Modell",
        ),
        ModelArtifact(
            url="https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/hed_edge_detection/deploy.prototxt",
            path=Path(models.get("hed", {}).get("proto", "hed_repo/deploy.prototxt")),
            description="HED Prototxt",
        ),
        ModelArtifact(
            url="https://github.com/csyanbin/DexiNed/releases/download/v1.0/dexined.pth",
            path=Path(models.get("dexined", "dexined_repo/weights/dexined.pth")),
            description="DexiNed Gewicht",
        ),
        ModelArtifact(
            url="1IQ9JgqGJjgpZAZTzrfC0YBv9l2nhVqLt",
            path=Path(models.get("casenet", "casenet_repo/model/casenet.pth")),
            description="CASENet Gewicht",
            is_gdrive=True,
        ),
        ModelArtifact(
            url="https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz",
            path=Path(models.get("structured_forest", "models/structured/model.yml.gz")),
            description="Structured Forest Modell",
        ),
    ]


def missing_artifacts(artifacts: Iterable[ModelArtifact]) -> List[ModelArtifact]:
    """Filtert alle fehlenden Artefakte."""

    return [artifact for artifact in artifacts if not artifact.path.exists()]


def ensure_all_models_available(config: Dict) -> None:
    """Stellt sicher, dass alle benötigten Modelle vorhanden sind.

    Fehlende Dateien werden automatisch heruntergeladen, damit die GUI
    ohne manuelle Vorbereitung lauffähig bleibt.
    """

    artifacts = build_artifacts_from_config(config)
    pending = missing_artifacts(artifacts)
    if not pending:
        logger.info("Alle Modelldateien sind bereits vorhanden.")
        return

    logger.info("%d Modelldatei(en) fehlen – starte Download.", len(pending))
    for artifact in pending:
        logger.info("Lade %s … (%s)", artifact.description, artifact.path)
        download_file(artifact.url, artifact.path, is_gdrive=artifact.is_gdrive)

    logger.info("Alle fehlenden Modelldateien wurden heruntergeladen.")
