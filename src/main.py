import glob, os
import re
import mne
import numpy as np
import pandas as pd
from features import periodograma_welch,PSD_banda,aperiodico,asimetria_alfa, ISA, resultado_asimetria, resultado_aperiodico,resultado_ISA,resultados_PSD
from preprocesamiento import pasa_banda,notch
import pandas as pd
from openpyxl import Workbook
from exportar_resultado import exportar_PSD, exportar_aperiodico, exportar_asimetria, exportar_ISA

#definimos variables
regiones = {
"frontal": ["Fz", "F1", "F2", "F3", "F4","F5","F6","F7","F8"],
"central": ["Cz", "C1", "C2", "C3", "C4","C5","C6"],
"parietal": ["Pz", "P1", "P2", "P3", "P4","P5","P6","P7","P8"],
"occipital": ["Oz", "O1", "O2"],
}


bandas = {
"delta": [1,4],
"theta": [4,8],
"alfa": [8,13],
"beta": [13,30],
}


for archivo in glob.glob("./data/*.bdf"):
    numero_sujeto = int(re.search(r'sub-(\d+)', archivo).group(1))
    if numero_sujeto < 60: 
        grupo = "CTR"
    else:
        grupo = "EXP"

    raw = mne.io.read_raw_bdf(input_fname=archivo, preload=True)
    for region, electrodos in regiones.items():
        eeg_channels = electrodos #ME QUEDO SOLO CON LOS CANALES QUE QUIERO DE CADA REGION 
        raw_eeg=raw.copy().pick(eeg_channels) 
        resultad_ISA = ISA(raw_eeg,numero_sujeto,grupo,region)
        #ACA APLICAR PREPROCESAMIENTO   
        eeg_filtrado = pasa_banda(raw_eeg)

        #Aplico filtro nothc para sacar ruido de linea >50Hz
        for canal in eeg_channels:
            eeg_preprocesado = notch(eeg_filtrado,eeg_channels)

            psds_f, freqs_f,psd_F3,psd_F4 = periodograma_welch(eeg_preprocesado,canal)

            for banda, rango in bandas.items():
                #PSD POR BANDA
                PSD_banda(banda,rango,freqs_f,psds_f,numero_sujeto,grupo,region,resultados_PSD)
                #analisis aperiodico
                aperiodico(freqs_f,psds_f,numero_sujeto,grupo,region,resultado_aperiodico)

    #ASIMETRIA AFLFA (F3,F4)
    asimetria_alfa(freqs_f, psd_F4,psd_F3,numero_sujeto,grupo,resultado_asimetria)


exportar_PSD(resultados_PSD)
exportar_aperiodico(resultado_aperiodico)
exportar_asimetria(resultado_asimetria)
exportar_ISA(resultado_ISA)

