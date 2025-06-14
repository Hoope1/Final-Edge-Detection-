
# 🧠 Modern Edge Detection Toolkit

Ein modernes, auf Deep Learning basiertes Toolkit zur Kantenerkennung in Bildern – vollautomatisch, vollständig in Python, ausschließlich mit GUI.

## 🚀 Features

- 6 moderne Kantendetektoren:
  - BDCN
  - RCF
  - HED
  - DexiNed
  - CASENet
  - Structured Forest (OpenCV contrib)
- Streamlit-GUI (kein CLI, keine Vorschau, keine Interaktion)
- GPU-Beschleunigung (CUDA)
- Batchverarbeitung für ganze Ordner
- Fortschrittsanzeige und Logging
- Einheitliches Ausgabeformat (invertiert, druckfähig)

## 🗂 Projektstruktur

```bash
modern_edge_toolkit/
├── gui_modern.py            # Streamlit Frontend
├── gui_core.py              # Verarbeitungslogik
├── detectors_modern.py      # Alle Detektor-Klassen
├── detector_base.py         # Gemeinsame Basisklasse
├── utils_image.py           # Hilfsfunktionen für Bilder
├── config_modern.yaml       # Zentrale Konfiguration
├── log_modern.py            # Logging-Setup
├── tests/                   # Tests für Detektoren
├── requirements.txt         # Abhängigkeiten
├── setup_script.bat         # Windows-Setup für Modelle
├── results/                 # Automatisch erzeugt
├── images/                  # Eingabebilder
├── bdcn_repo/ (submodule)
├── rcf_repo/  (submodule)
├── hed_repo/  (submodule)
├── dexined_repo/ (submodule)
├── casenet_repo/ (submodule)
└── models/structured/
```

## ⚙️ Installation

```bash
git clone --recurse-submodules https://github.com/<user>/modern_edge_toolkit.git
cd modern_edge_toolkit
python -m venv venv
source venv/bin/activate  # oder .\venv\Scripts\activate
pip install -r requirements.txt
```

## 📥 Modelle installieren

```bash
setup_script.bat
```

Falls `gdown` fehlt: `pip install gdown`

## 🧪 Starten

```bash
streamlit run gui_modern.py
```

## 🖼 Ergebnisse

Alle verarbeiteten Bilder werden als PNG gespeichert unter:

```
results/<method>/<original_name>.png
```

## 👤 Zielgruppe

- Forschende im Bereich Computer Vision
- Experimente mit Kantendetektoren
- Downstream-Preprocessing für Segmentierung/Matching
