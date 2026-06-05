import mne
import numpy as np

def pasa_banda(raw_eeg):
    #aplicamos pasabanda 0.5 - 35 Hz
    eeg_filtrado=raw_eeg.filter(l_freq=0.5, h_freq=35)
    
    return eeg_filtrado

def notch(eeg_filtrado,eeg_channels):
    for canal in eeg_channels:
        #region = eeg_channels[p]
        eeg_preprocesado = mne.filter.notch_filter(
            eeg_filtrado.get_data(picks=canal)[0],
            Fs=1024,
            freqs=[16,32,49],
            notch_widths = 2,
            method = 'fir'
        )

    return eeg_preprocesado