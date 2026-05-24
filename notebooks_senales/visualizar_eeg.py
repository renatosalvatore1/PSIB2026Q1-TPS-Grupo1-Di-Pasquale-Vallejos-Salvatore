"""
=============================================================
  Visualizador de señales EEG — Dataset ds003969
  Proyecto: Comparación EEG meditadores vs controles
  Materia: Procesamiento de Señales Biomédicas
=============================================================

INSTALACIÓN DE DEPENDENCIAS (ejecutar una vez):
    pip install mne matplotlib numpy

USO:
    python visualizar_eeg.py <ruta_al_bdf> [opciones]

EJEMPLOS:
    # Visualización estándar (filtro 0.5-40 Hz, épocas de 30s)
    python visualizar_eeg.py sub-080/eeg/sub-080_task-med2_eeg.bdf

    # Ver actividad infra-lenta (banda ISA del proyecto)
    python visualizar_eeg.py sub-080/eeg/sub-080_task-med2_eeg.bdf --filtro isa

    # Sin filtrar (señal DC original del BioSemi)
    python visualizar_eeg.py sub-080/eeg/sub-080_task-med2_eeg.bdf --filtro raw

    # Mostrar más canales y también EOG + fisiológicos
    python visualizar_eeg.py sub-080/eeg/sub-080_task-med2_eeg.bdf --n-canales 30 --todos-canales

    # Cambiar duración de época
    python visualizar_eeg.py sub-080/eeg/sub-080_task-med2_eeg.bdf --duracion 60

NAVEGACIÓN EN EL VISOR:
    →  /  ←       avanzar / retroceder una época
    +  /  -       aumentar / reducir la escala de amplitud
    j             ir a un tiempo específico
    a             toggle de anotaciones
    clic en canal ocultar/mostrar canal individual
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import mne

mne.set_log_level('WARNING')


# ──────────────────────────────────────────────
#  1. CARGA DEL ARCHIVO BDF
# ──────────────────────────────────────────────

def cargar_bdf(ruta):
    """
    Lee un archivo .bdf (BioSemi ActiveTwo) del dataset.
    El dataset fue grabado a 1024 Hz con DC coupling (sin filtro de paso alto
    en hardware), lo que permite analizar actividad infra-lenta (ISA).
    """
    if not os.path.isfile(ruta):
        sys.exit(f"\n[ERROR] No se encontró el archivo: {ruta}\n"
                 f"        Verificá que la ruta sea correcta.\n")

    print(f"\nCargando: {os.path.basename(ruta)}")
    raw = mne.io.read_raw_bdf(ruta, preload=True, verbose=False)
    print(f"  Canales totales : {len(raw.ch_names)}")
    print(f"  Frecuencia      : {raw.info['sfreq']} Hz")
    print(f"  Duración        : {raw.times[-1]:.1f} s  ({raw.times[-1]/60:.1f} min)")
    return raw


# ──────────────────────────────────────────────
#  2. CONFIGURACIÓN DE TIPOS DE CANALES
# ──────────────────────────────────────────────

def configurar_canales(raw):
    """
    Asigna el tipo correcto a cada canal del dataset ds003969.

    El setup BioSemi del dataset incluye:
      - 64 canales EEG  (Fp1 ... O2, sistema 10-20 extendido)
      - VEOG, HEOG      (electrooculograma vertical y horizontal)
      - EXG3 ... EXG8   (señales fisiológicas: Resp, ECG, GSR, mastoides)
      - Status          (canal de triggers/eventos)

    Referencia: mastoid derecho (EXG2 en BioSemi).
    """
    tipos = {}
    for ch in raw.ch_names:
        ch_upper = ch.upper()
        if ch_upper in ('VEOG', 'HEOG'):
            tipos[ch] = 'eog'
        elif ch_upper.startswith('EXG') or ch_upper in ('RESP', 'ECG', 'GSR',
                                                          'MASTL', 'MASTR'):
            tipos[ch] = 'misc'
        elif ch_upper == 'STATUS':
            tipos[ch] = 'stim'

    if tipos:
        raw.set_channel_types(tipos)

    # Aplicar montaje BioSemi64 para tener coordenadas de electrodos
    # (necesario para topografías, no para la visualización en tira)
    try:
        montaje = mne.channels.make_standard_montage('biosemi64')
        raw.set_montage(montaje, on_missing='ignore', verbose=False)
    except Exception:
        pass  # Si los nombres no coinciden exactamente, continuar igual

    # Re-referenciar al promedio (la referencia original es el mastoid derecho)
    # Para análisis ISA Sihn 2024 usa Laplaciano superficial, pero para
    # visualización la referencia promedio es suficiente.
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) > 0:
        raw.set_eeg_reference('average', projection=False, verbose=False)

    return raw


# ──────────────────────────────────────────────
#  3. FILTRADO
# ──────────────────────────────────────────────

def filtrar(raw, tipo='visualizacion'):
    """
    Aplica filtrado según el tipo de análisis que se quiera inspeccionar.

    'visualizacion'  0.5 – 40 Hz   Oscilaciones típicas (delta a gamma bajo).
                                    Bueno para inspeccionar calidad de señal,
                                    artefactos, alpha, theta, etc.

    'isa'            0.01 – 0.1 Hz  Banda infra-lenta (ISA). La banda exacta
                                    del proyecto es 0.03-0.08 Hz (Sihn 2024).
                                    Necesitás épocas largas para ver oscilaciones.

    'raw'            Sin filtrar    Señal DC tal como la grabó el BioSemi.
                                    Útil para verificar derivas lentas, nivel DC.
    """
    r = raw.copy()
    if tipo == 'visualizacion':
        print("Filtro: 0.5 – 40 Hz (FIR, Hamming)")
        r.filter(l_freq=0.5, h_freq=40.0, method='fir',
                 fir_window='hamming', verbose=False)
    elif tipo == 'isa':
        print("Filtro: 0.01 – 0.1 Hz (banda ISA, FIR, Hamming)")
        # Para filtros tan bajos se necesita una ventana muy larga;
        # MNE la calcula automáticamente.
        r.filter(l_freq=0.01, h_freq=0.1, method='fir',
                 fir_window='hamming', verbose=False)
    elif tipo == 'raw':
        print("Sin filtrado (señal DC original)")
    return r


# ──────────────────────────────────────────────
#  4. VISUALIZACIÓN
# ──────────────────────────────────────────────

ESCALAS_DEFAULT = {
    'visualizacion' : dict(eeg=75e-6,  eog=200e-6, misc=1.0),
    'isa'           : dict(eeg=300e-6, eog=500e-6, misc=1.0),
    'raw'           : dict(eeg=200e-6, eog=500e-6, misc=1.0),
}

def visualizar(raw, duracion=30.0, n_canales=20,
               filtro_tipo='visualizacion', titulo=None, solo_eeg=True,
               canales=None):
    """
    Abre el visor interactivo de MNE con épocas de la duración especificada.

    Parámetros
    ----------
    duracion   : float      — segundos por ventana (default 30)
    n_canales  : int        — canales visibles simultáneamente (default 20)
    filtro_tipo: str        — para elegir la escala de amplitud adecuada
    titulo     : str        — texto del título de la ventana
    solo_eeg   : bool       — True = solo canales EEG; False = incluye EOG y misc
    canales    : list[str]  — lista de nombres de canales a mostrar (ej. ['C1'])
    """
    if canales:
        picks = mne.pick_channels(raw.info['ch_names'], include=canales)
    elif solo_eeg:
        picks = 'eeg'
    else:
        picks = None

    escala = ESCALAS_DEFAULT.get(filtro_tipo, ESCALAS_DEFAULT['visualizacion'])

    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    n_mostrar = len(canales) if canales else min(n_canales, n_eeg)

    print(f"\n{'─'*55}")
    print(f"  Canales EEG disponibles : {n_eeg}")
    print(f"  Canales mostrados       : {n_mostrar}")
    print(f"  Duración de época       : {duracion} s")
    print(f"{'─'*55}")
    print("  NAVEGACIÓN")
    print("    →  /  ←       siguiente / anterior época")
    print("    +  /  -       ampliar / reducir amplitud")
    print("    j             ir a tiempo específico")
    print("    clic en canal ocultar/mostrar ese canal")
    print(f"{'─'*55}\n")

    raw.plot(
        duration    = duracion,
        n_channels  = n_mostrar,
        scalings    = escala,
        title       = titulo or "EEG",
        picks       = picks,
        show_scrollbars = True,
        show_scalebars  = True,
        block       = True,
        verbose     = False,
    )


# ──────────────────────────────────────────────
#  5. ENTRADA POR LÍNEA DE COMANDO
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualizador EEG — Dataset ds003969 (meditación vs pensamiento)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ejemplos:
  python visualizar_eeg.py sub-080/eeg/sub-080_task-med2_eeg.bdf
  python visualizar_eeg.py sub-080/eeg/sub-080_task-med2_eeg.bdf --filtro isa
  python visualizar_eeg.py sub-080/eeg/sub-080_task-think1_eeg.bdf --filtro raw --duracion 60
        """
    )

    parser.add_argument(
        'archivo',
        help='Ruta al archivo .bdf del dataset BIDS.'
    )
    parser.add_argument(
        '--filtro',
        choices=['visualizacion', 'isa', 'raw'],
        default='visualizacion',
        metavar='TIPO',
        help=(
            '"visualizacion" (0.5-40 Hz, default)  |  '
            '"isa" (0.01-0.1 Hz)  |  '
            '"raw" (sin filtrar)'
        )
    )
    parser.add_argument(
        '--duracion',
        type=float,
        default=30.0,
        metavar='SEG',
        help='Segundos por época en el visor (default: 30)'
    )
    parser.add_argument(
        '--n-canales',
        type=int,
        default=20,
        metavar='N',
        help='Canales visibles simultáneamente (default: 20)'
    )
    parser.add_argument(
        '--todos-canales',
        action='store_true',
        help='Incluir EOG y canales fisiológicos (por default solo EEG)'
    )
    parser.add_argument(
        '--canales',
        nargs='+',
        metavar='CANAL',
        help='Mostrar solo estos canales (ej. --canales C1 C2 Fz)'
    )

    args = parser.parse_args()

    # ── Pipeline ──────────────────────────────
    raw  = cargar_bdf(args.archivo)
    raw  = configurar_canales(raw)
    raw  = filtrar(raw, tipo=args.filtro)

    nombre = os.path.basename(args.archivo).replace('_eeg.bdf', '')
    titulo = f"{nombre}   |   filtro: {args.filtro}   |   época: {args.duracion}s"

    visualizar(
        raw,
        duracion   = args.duracion,
        n_canales  = args.n_canales,
        filtro_tipo= args.filtro,
        titulo     = titulo,
        solo_eeg   = not args.todos_canales,
        canales    = args.canales,
    )


if __name__ == '__main__':
    main()
