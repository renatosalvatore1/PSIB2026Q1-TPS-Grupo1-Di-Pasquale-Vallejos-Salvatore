import matplotlib.pyplot as plt 
import mne
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
filename = 'sub-029_task-med2_eeg.bdf'
raw = mne.io.read_raw_bdf(DATA_DIR / filename, preload=True)
raw_eeg = raw.copy().pick_types(eeg=True, exclude='bads') #solo canales EEG

raw_og = raw_eeg.copy()

#filtro para ruido de línea 50 Hz by nato
raw_eeg.notch_filter(freqs=np.arange(48, 512, 32),method = 'fir')
raw_notch = raw_eeg.copy()

#pasa-banda ya que EMG ocupa bandas mayores a 40 Hz
raw_eeg.filter(l_freq=0.5, h_freq=40.0)

fig, ax = plt.subplots(figsize= (10,5))

#ORIGINAL
psd = raw_og.compute_psd(method='welch', fmin=0.5, fmax=50)
psds, freqs = psd.get_data(return_freqs=True)
ax.plot(freqs, np.mean(10 * np.log10(psds), axis=0),color='blue',alpha=0.7, label='Señal original')

#FILTRADA RUIDO DE LÍNEA
notch = raw_notch.compute_psd(method='welch', fmin=0.5, fmax=50)
psdN, freqsN = notch.get_data(return_freqs=True)
ax.plot(freqsN, np.mean(10 * np.log10(psdN), axis=0),color='red',alpha=0.7,label='Sin ruido de línea')

#FILTRADA RUIDO EMG
emg = raw_eeg.compute_psd(method='welch', fmin=0.5, fmax=50)
psdM, freqsM = emg.get_data(return_freqs=True)
ax.plot(freqsM, np.mean(10 * np.log10(psdM), axis=0),color='green',alpha=0.7,label='Sin ruido EMG')

ax.set_xlabel('Frecuencia (Hz)')
ax.set_ylabel('PSD (dB/Hz)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()