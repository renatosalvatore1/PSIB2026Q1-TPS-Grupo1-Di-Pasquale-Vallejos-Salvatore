import os
import gdown
from pathlib import Path
import mne
import matplotlib.pyplot as plt
import numpy as np

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

raw = mne.io.read_raw_bdf(input_fname='./data/sub-050_task-med2_eeg.bdf', preload=True)
eeg_channels = raw.ch_names[:64]
#Me quedo solo con los EEG
raw_eeg=raw.copy().pick(eeg_channels)

psds, freqs = mne.time_frequency.psd_array_welch(
    raw_eeg.get_data(), 
    sfreq=1024, 
    n_fft=4096,
    n_per_seg=4096, 
    output="power")


plt.title("PSD del canal 30 sin filtrar")
plt.plot(freqs,psds[30])
plt.semilogy()
plt.xlim(0,500)
plt.show()

#Aplico filtro nothc para sacar ruido de linea >50Hz

filtrado = mne.filter.notch_filter(
    raw_eeg.get_data()[30],
    Fs=1024,
    freqs=np.arange(48, 512, 32),
    notch_widths = None,
    method = 'fir'
)

psds_f, freqs_f = mne.time_frequency.psd_array_welch(
    filtrado, 
    sfreq=1024, 
    n_fft=4096,
    n_per_seg=4096, 
    output="power")
plt.title("PSD del canal 30 con filtro notch")
plt.plot(freqs_f, psds_f)
plt.semilogy()
plt.xlim(0,500)
plt.show()



