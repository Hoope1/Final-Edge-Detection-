
@echo off
setlocal enabledelayedexpansion

echo.
echo ===============================================
echo   Modern Edge Detection Toolkit – SETUP SCRIPT
echo ===============================================
echo.

REM Python & curl prüfen
python --version >nul 2>&1 || (
    echo [FEHLER] Python nicht gefunden. Bitte Python installieren.
    pause
    exit /b 1
)
curl --version >nul 2>&1 || (
    echo [FEHLER] 'curl' nicht gefunden. Bitte in PATH aufnehmen.
    pause
    exit /b 1
)

REM Verzeichnisse erstellen
mkdir bdcn_repo\pretrained 2>nul
mkdir rcf_repo\model 2>nul
mkdir dexined_repo\weights 2>nul
mkdir casenet_repo\model 2>nul
mkdir models\structured 2>nul
mkdir hed_repo 2>nul

REM Submodule initialisieren
echo [1/6] Initialisiere Submodule...
git submodule update --init --recursive

REM Downloads starten
echo [2/6] Lade Modelle herunter...

echo   -> HED Weights
curl -L -o hed_repo\hed_pretrained_bsds.caffemodel https://github.com/s9xie/hed/raw/master/examples/hed/hed_pretrained_bsds.caffemodel
curl -L -o hed_repo\deploy.prototxt https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/hed_edge_detection/deploy.prototxt

echo   -> BDCN Weights
curl -L -o bdcn_repo\pretrained\bdcn_pretrained.pth https://github.com/zijundeng/BDCN/releases/download/v1.0.0/bdcn_pretrained.pth

echo   -> RCF Weights (Google Drive)
gdown 1qxW3Z4Y6z3dpZJkZHZbwAb29rT1U3pS8 -O rcf_repo\model\RCF.pth || echo [WARNUNG] Bitte manuell herunterladen.

echo   -> DexiNed Weights
curl -L -o dexined_repo\weights\dexined.pth https://github.com/csyanbin/DexiNed/releases/download/v1.0/dexined.pth

echo   -> CASENet Weights (Google Drive)
gdown 1IQ9JgqGJjgpZAZTzrfC0YBv9l2nhVqLt -O casenet_repo\model\casenet.pth || echo [WARNUNG] Bitte manuell herunterladen.

echo   -> Structured Forest Model
curl -L -o models\structured\model.yml.gz https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz

echo.
echo ===============================================
echo ✅ SETUP abgeschlossen!
echo 📁 Modelle befinden sich jetzt in den Zielordnern.
echo ❗ Prüfe manuell, ob .pth / .caffemodel / .yml.gz Dateien vorhanden sind.
echo ===============================================
echo.
pause
