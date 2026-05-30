import os
import gdown
from pathlib import Path
import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mne.viz import plot_filter, plot_ideal_filter

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

raw = mne.io.read_raw_bdf(input_fname='./data/sub-070_task-med2_eeg.bdf', preload=True)
eeg_channels = raw.ch_names[:64]
#Me quedo solo con los EEG
raw_eeg=raw.copy().pick(eeg_channels)

#aplicamos pasabajos con fc=100 Hz
eeg_filtrado=raw_eeg.filter(l_freq=None, h_freq=100)

psds, freqs = mne.time_frequency.psd_array_welch(
    eeg_filtrado.get_data(), 
    sfreq=1024, 
    n_fft=4096,
    fmax=100,
    n_per_seg=4096, 
    output="power")

a= np.arange(49, 512, 32) 
b= np.arange(49,512,50)
freqs_notch = (np.concatenate((a,[49.7]), axis=None))
plt.plot(freqs,psds[30])
plt.semilogy()
plt.xlim(0,100)
plt.show()

#Aplico filtro nothc para sacar ruido de linea >50Hz

filtrado = mne.filter.notch_filter(
    eeg_filtrado.get_data()[30],
    Fs=1024,
    freqs=a,
    notch_widths = 2,
    method = 'fir'
)

psds_f, freqs_f = mne.time_frequency.psd_array_welch(
    filtrado, 
    sfreq=1024, 
    n_fft=4096,
    fmax=100,
    n_per_seg=4096, 
    output="power")

plt.plot(freqs_f, psds_f)
plt.loglog()
plt.xlim(0,100)
plt.show()

#ANALISIS PSD POR
bandas = [[1,4],[4, 8], [8,13],[13,30]]

for i in range(len(bandas)):
    mask = (freqs_f >= bandas[i][0]) & (freqs_f <= bandas[i][1])
    band_power = np.trapezoid(psds_f[mask],freqs_f[mask])
    print(band_power)




