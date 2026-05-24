import os
import gdown
from pathlib import Path

# ─── Configuración ────────────────────────────────────────────────
FOLDER_URL = "https://drive.google.com/drive/folders/1fzN5R6Bzm3g0PjLOplewveW8wo181J60?usp=drive_link"
DATA_DIR = Path("data")  # carpeta destino (relativa al cwd)

# ─── Descarga ─────────────────────────────────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)

if not any(DATA_DIR.iterdir()):
    print(f"Descargando archivos a {DATA_DIR.resolve()}...")
    gdown.download_folder(
        FOLDER_URL,
        output=str(DATA_DIR),
        quiet=False,
        use_cookies=False,
        
    )
    print("✓ Descarga completa")
else:
    n = len(list(DATA_DIR.iterdir()))
    print(f"✓ {DATA_DIR} ya tiene {n} archivo(s), salteando descarga.")