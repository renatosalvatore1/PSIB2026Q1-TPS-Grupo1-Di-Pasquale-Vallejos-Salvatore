import mne
import numpy as np

def pasa_banda(raw_eeg):
    #aplicamos pasabanda 0.5 - 100 Hz
    eeg_filtrado=raw_eeg.filter(l_freq=0.5, h_freq=100)
    return eeg_filtrado
    

    
    


def notch(eeg_filtrado,eeg_channels):
    freqs_notch= np.arange(49, 100, 32) 
    for canal in eeg_channels:
        #region = eeg_channels[p]
        eeg_preprocesado = mne.filter.notch_filter(
            eeg_filtrado.get_data(picks=canal)[0],
            Fs=1024,
            freqs=freqs_notch,
            notch_widths = 2,
            method = 'fir'
        )

    return eeg_preprocesado