import mne
import numpy as np
from specparam import SpectralModel

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

    if canal in ["F3"]:
        psd_F3 = psds_f
    if canal in ["F4"]:
        psd_F4 = psds_f

    return psds_f, freqs_f, psd_F3, psd_F4


def PSD_banda(banda, rango, freqs_f, psds_f, numero_sujeto, grupo, region, resultados_PSD):
    mask = (freqs_f >= rango[0]) & (freqs_f <= rango[1])
    potencia = np.trapezoid(psds_f[mask], freqs_f[mask])

    resultados_PSD.append({
        "sujeto": numero_sujeto,
        "grupo": grupo,
        "region": region,
        "banda": banda,
        "potencia": potencia,
    })


def aperiodico(freqs_f, psds_f, numero_sujeto, grupo, region, resultado_aperiodico):
    freq_mask = (freqs_f >= 0.5) & (freqs_f <= 50)
    freqs_crop = freqs_f[freq_mask]
    psds_crop = psds_f[freq_mask]

    fm = SpectralModel(min_peak_height=0.1, verbose=False)
    fm.fit(freqs_crop, psds_crop)

    resultado_aperiodico.append({
        "sujeto": numero_sujeto,
        "grupo": grupo,
        "region": region,
        "aperiodico": fm.results.params.aperiodic.params[1],
    })


def asimetria_alfa(freqs_f, psd_F4, psd_F3, numero_sujeto, grupo, resultado_asimetria):
    if psd_F3 is not None and psd_F4 is not None:
        mask = (freqs_f >= 8) & (freqs_f <= 13)
        potencia_F4 = np.trapezoid(psd_F4[mask], freqs_f[mask])
        potencia_F3 = np.trapezoid(psd_F3[mask], freqs_f[mask])
        asimetria = np.log(potencia_F4) - np.log(potencia_F3)
        resultado_asimetria.append({
            "sujeto": numero_sujeto,
            "grupo": grupo,
            "asimetria": asimetria,
        })


def ISA(señal_sujeto, numero_sujeto, grupo, region, resultado_ISA):
    """
    Pipeline ISA (Sihn et al. 2024):
    - downsample a 8 Hz
    - filtro FIR Hamming 0.03-0.08 Hz
    - Hilbert -> envolvente de amplitud
    - mediana temporal (robusta a outliers)
    Guarda un resultado por canal.
    """
    raw_isa = señal_sujeto.copy()
    raw_isa.resample(sfreq=8)
    raw_isa.filter(l_freq=0.03, h_freq=0.08, method='fir', fir_window='hamming')
    raw_isa.apply_hilbert(envelope=True)
    data = raw_isa.get_data()  # shape: (n_canales, n_muestras)
    mediana_isa = np.median(data, axis=1)  # mediana por canal

    for i, canal in enumerate(raw_isa.ch_names):
        resultado_ISA.append({
            "sujeto": numero_sujeto,
            "grupo": grupo,
            "region": region,
            "canal": canal,
            "mediana_isa": mediana_isa[i],
        })

    return resultado_ISA