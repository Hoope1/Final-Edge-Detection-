
import streamlit as st
import os
import logging
from gui_core import load_config, process_images
from log_modern import setup_logger

# Initialisiere Logger
setup_logger()

# Seite konfigurieren
st.set_page_config(
    page_title="Modern Edge Detection Toolkit",
    layout="wide"
)

# Lade Konfiguration
config = load_config()

st.title("🔍 Modern Edge Detection Toolkit – Deep & GUI-only")
st.markdown("Wähle Eingabeverzeichnis und Detektionsmethoden:")

# Auswahl Eingabeordner (optional, falls manuelle Auswahl erlaubt werden soll)
image_dir = st.text_input("📁 Bildordner", value=config.get("image_dir", "images"))
config["image_dir"] = image_dir

# Methoden-Auswahl
method_options = {
    "BDCN": "bdcn",
    "RCF": "rcf",
    "HED": "hed",
    "DexiNed": "dexined",         # Noch zu implementieren
    "CASENet": "casenet",         # Noch zu implementieren
    "Structured Forest": "structured_forest"
}

selected_methods = st.multiselect(
    "🧠 Wähle Detektions-Algorithmen:",
    options=list(method_options.keys()),
    default=["BDCN", "HED"]
)

method_keys = [method_options[m] for m in selected_methods]

# Startbutton
if st.button("🚀 Verarbeitung starten"):
    st.info("Starte Verarbeitung... bitte warten.")
    progress_bar = st.progress(0)
    from time import sleep

    # Callback-Wrapper für Fortschritt
    def tqdm_streamlit(iterable, total, desc):
        for i, item in enumerate(iterable):
            yield item
            progress_bar.progress((i + 1) / total)
        progress_bar.empty()

    import gui_core
    original_tqdm = gui_core.tqdm
    gui_core.tqdm = lambda *args, **kwargs: tqdm_streamlit(*args, **kwargs)

    results = process_images(method_keys, config)

    gui_core.tqdm = original_tqdm
    st.success("✅ Verarbeitung abgeschlossen!")

    for method, files in results.items():
        st.markdown(f"### 🖼 Ergebnisse: {method.upper()}")
        for f in files:
            st.markdown(f"- {os.path.basename(f)} gespeichert in `{os.path.dirname(f)}`")
