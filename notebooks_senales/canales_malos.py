import matplotlib.pyplot as plt 
import mne
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
filename = 'sub-029_task-med2_eeg.bdf'
raw = mne.io.read_raw_bdf(DATA_DIR / filename, preload=True)

def detectar_canales_malos(raw: mne.io.Raw) -> list[str]:
    """
    Estrategia simple basada en z-score de la varianza por canal.
    Umbral: (UMBRAL_BAD_CH = 5) desviaciones estándar sobre la media.
    """
    picks_eeg = mne.pick_types(raw.info, eeg=True, exclude='bads')
    data      = raw.get_data(picks=picks_eeg)

    var       = np.var(data, axis=1) #Varianza por canal
    z_scores  = (var - np.mean(var)) / np.std(var)

    malos = [raw.ch_names[picks_eeg[i]]
             for i, z in enumerate(z_scores)
             if abs(z) > 5] #UMBRAL_BAD_CH = 5

    if malos:
        print(f"    Canales malos detectados: {malos}")
    else:
        print("    No se detectaron canales malos")

    return malos

def eliminar_malos(raw: mne.io.Raw, malos: list[str]) -> mne.io.Raw:
    """
    Elimina los canales malos del objeto Raw.
    Como no vamos a hacer ICA, por ejemplo, elimino en vez de interpolar 
    """
    if not malos:
        return raw
    raw.drop_channels(malos)
    print(f"    Canales eliminados: {malos}")
    return raw

malos = detectar_canales_malos(raw)
raw_bueno = eliminar_malos(raw, malos)


fig, ax = plt.subplots(figsize= (10,5))

psd_og = raw.compute_psd(method='welch', fmin=0.5, fmax=50)
psds_og, freqs_og = psd_og.get_data(return_freqs=True)
ax.plot(freqs_og, np.mean(10 * np.log10(psds_og), axis=0), color='red', alpha=0.7, label='Con canales malos')
#for psd_canal in psds_og:
    #ax.plot(freqs_og, 10*np.log10(psd_canal),color='red',alpha=0.1,linewidth=0.5)

psd = raw.compute_psd(method='welch', fmin=0.5, fmax=50)
psds, freqs = psd.get_data(return_freqs=True)
ax.plot(freqs, np.mean(10 * np.log10(psds), axis=0),color='blue', alpha=0.7, label='Sin canales malos')
#for psd_canal in psds:
    #ax.plot(freqs, 10*np.log10(psd_canal),color='black',alpha=0.1,linewidth=0.5)

ax.set_xlabel('Frecuencia (Hz)')
ax.set_ylabel('PSD (dB/Hz)')
ax.set_title('Comparación PSD — promedio sobre canales')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
