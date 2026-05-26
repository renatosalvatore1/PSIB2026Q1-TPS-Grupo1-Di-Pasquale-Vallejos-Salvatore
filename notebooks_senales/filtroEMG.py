import matplotlib.pyplot as plt 
import mne
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
filename = 'sub-029_task-med2_eeg.bdf'
raw = mne.io.read_raw_bdf(DATA_DIR / filename, preload=True)

'''
Como se puede observar en "canales_malos.py", el análisis en frecuencia con 
canales malos o sin canales malos no demuestra una diferencia significativa.
Por lo tanto, se trabajará con TODOS los canales.
'''







fig, ax = plt.subplots(figsize= (10,5))

psd = raw.compute_psd(method='welch', fmin=0.5, fmax=50)
psds, freqs = psd.get_data(return_freqs=True)
ax.plot(freqs, np.mean(10 * np.log10(psds), axis=0),color='blue')

ax.set_xlabel('Frecuencia (Hz)')
ax.set_ylabel('PSD (dB/Hz)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()