import mne
import numpy as np
import numpy as np
from specparam import SpectralModel
import numpy as np

resultados_PSD = []
resultado_aperiodico = []
resultado_asimetria = []
resultado_ISA = []

def periodograma_welch(eeg_preprocesado, canal):
    psd_F3 = None
    psd_F4 = None
    psds_f, freqs_f = mne.time_frequency.psd_array_welch(
        eeg_preprocesado, 
        sfreq=1024, 
        n_fft=4096,
        fmax=100,
        n_per_seg=4096, 
        output="power")
    
    #me guardo los psds para calcular la asimetria despues
    if canal in ["F3"]:
                psd_F3 = psds_f
    if canal in ["F4"]:
                psd_F4 = psds_f

    return psds_f, freqs_f, psd_F3, psd_F4


def PSD_banda(banda, rango,freqs_f,psds_f,numero_sujeto,grupo,region,resultados_PSD):
    #resultados_PSD = []
    mask = (freqs_f >= rango[0]) & (freqs_f <= rango[1])
    potencia = np.trapezoid(psds_f[mask],freqs_f[mask])

    resultados_PSD.append({
        "sujeto": numero_sujeto,
        "grupo": grupo,
        "region": region,
        "banda": banda,
        "potencia": potencia,
        })    
    

def aperiodico(freqs_f,psds_f, numero_sujeto,grupo,region,resultado_aperiodico):
    #resultado_aperiodico=[]
    sm = SpectralModel(verbose=False)
    freq_mask = (freqs_f >= 0.5) & (freqs_f <= 50)
    freqs_crop = freqs_f[freq_mask]
    psds_crop  = psds_f[freq_mask]

    # Initialize model object and fit power spectrum
    fm = SpectralModel(min_peak_height=0.1)
    fm.fit(freqs_crop, psds_crop)

    resultado_aperiodico.append({
    "sujeto": numero_sujeto,
    "grupo": grupo,
    "region": region,
    "aperiodico": fm.results.params.aperiodic.params[1],
    })    

    

def asimetria_alfa(freqs_f, psd_F4,psd_F3,numero_sujeto,grupo,resultado_asimetria):
    if psd_F3 is not None and psd_F4 is not None:
        #resultado_asimetria = []
        mask = (freqs_f >= 8) & (freqs_f <= 13)
        potencia_F4 = np.trapezoid(psd_F4[mask],freqs_f[mask])
        potencia_F3 = np.trapezoid(psd_F3[mask],freqs_f[mask])
        asimetria = np.log(potencia_F4) - np.log(potencia_F3)
        resultado_asimetria.append({
                    "sujeto": numero_sujeto,
                    "grupo": grupo,
                    "asimetria": asimetria,
                    })
        
def ISA(señal_sujeto, numero_sujeto, grupo,region):
      raw_isa = señal_sujeto.copy()
      raw_isa.resample(sfreq=8)   
      raw_isa.filter(l_freq=0.03, h_freq=0.08, method='fir', fir_window='hamming')
      raw_isa.apply_hilbert(envelope=True)
      data= raw_isa.get_data()
      mediana_isa = np.median(data, axis=1)
      resultado_ISA.append({
                    "sujeto": numero_sujeto,
                    "grupo": grupo,
                    "region": region,
                    "mediana": mediana_isa,
                    })
      
      return resultado_ISA



   