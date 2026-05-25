"""
=============================================================
  PREPROCESAMIENTO EEG — Meditadores vs Controles
  Dataset: ds003969 (Braboszcz et al. / Sihn et al. 2024)
  Materia: Procesamiento de Señales Biomédicas
=============================================================

PIPELINE (basado en Sihn et al. 2024):
  1. Cargar .bdf (BioSemi, 64 canales, 1024 Hz, DC-coupled)
  2. Asignar tipos de canales (EEG / EOG / misc / stim)
  3. Recortar últimos 10 min (la parte que importa según el paper)
  4. Filtro pasa-banda 0.5–45 Hz (FIR Hamming)
  5. Re-referencia al promedio
  6. Detección y eliminación de canales malos
  7. Downsample a 256 Hz (suficiente para análisis hasta 45 Hz)
  8. Guardar objeto Raw limpio como .fif

USO:
  python preprocesamiento_eeg.py

DEPENDENCIAS:
  pip install mne numpy matplotlib scipy
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')           # sin ventana gráfica (para servidor/Colab)
import matplotlib.pyplot as plt
import mne
from pathlib import Path

mne.set_log_level('WARNING')


# ══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — editá estos paths según tu estructura
# ══════════════════════════════════════════════════════════════

DATA_DIR    = Path("data")          # carpeta con los .bdf
OUTPUT_DIR  = Path("procesado")     # donde se guardan los .fif limpios
FIGURES_DIR = Path("figuras")       # plots de control de calidad

# Tus sujetos separados por grupo
GRUPOS = {
    "CTR": [
        "sub-029", "sub-031", "sub-035", "sub-037", "sub-039",
        "sub-043", "sub-046", "sub-048", "sub-050", "sub-052", "sub-054"
    ],
    "EXP": [
        "sub-060", "sub-061", "sub-065", "sub-066", "sub-070",
        "sub-071", "sub-073", "sub-074", "sub-075", "sub-078"
    ]
}

TAREA = "task-med2"   # el archivo que usamos: meditación sesión 2

# Parámetros de preprocesamiento
SFREQ_TARGET    = 256     # Hz de destino tras downsample
L_FREQ          = 0.5     # Hz — corte inferior del filtro
H_FREQ          = 45.0    # Hz — corte superior del filtro
DURACION_MIN    = 10 * 60 # segundos — últimos 10 min (como en el paper)
UMBRAL_BAD_CH   = 5.0     # z-score para marcar canal como malo


# ══════════════════════════════════════════════════════════════
#  PASO 1 — CARGAR ARCHIVO BDF
# ══════════════════════════════════════════════════════════════

def cargar_bdf(sujeto: str) -> mne.io.Raw | None:
    """
    Busca el .bdf del sujeto y lo carga con preload=True.
    Devuelve None si el archivo no existe.
    """
    nombre = f"{sujeto}_{TAREA}_eeg.bdf"
    ruta   = DATA_DIR / nombre

    # Algunos datasets usan sub-carpeta BIDS: data/sub-029/eeg/
    if not ruta.exists():
        ruta = DATA_DIR / sujeto / "eeg" / nombre

    if not ruta.exists():
        print(f"  [SKIP] No se encontró: {nombre}")
        return None

    print(f"  Cargando {nombre}...")
    raw = mne.io.read_raw_bdf(str(ruta), preload=True, verbose=False)
    print(f"    {len(raw.ch_names)} canales | {raw.info['sfreq']} Hz | "
          f"{raw.times[-1]/60:.1f} min")
    return raw


# ══════════════════════════════════════════════════════════════
#  PASO 2 — TIPOS DE CANALES (específico para BioSemi ds003969)
# ══════════════════════════════════════════════════════════════

def asignar_tipos(raw: mne.io.Raw) -> mne.io.Raw:
    """
    El dataset usa esta convención de canales externos:
      EXG1 = ojo izquierdo (HEOG-)   EXG2 = ojo derecho (HEOG+)
      EXG3 = ceja ojo izq (VEOG+)    EXG4 = debajo ojo izq (VEOG-)
      EXG5 = mastoides izq (M1)      EXG6 = mastoides der (M2)
      EXG7 = ECG (collar)            EXG8 = Fp1 extra
      Status = canal de triggers
    """
    tipos = {}
    for ch in raw.ch_names:
        cu = ch.upper()
        if cu in ('EXG1', 'EXG2'):
            tipos[ch] = 'eog'   # HEOG
        elif cu in ('EXG3', 'EXG4'):
            tipos[ch] = 'eog'   # VEOG
        elif cu in ('EXG5', 'EXG6', 'EXG7', 'EXG8'):
            tipos[ch] = 'misc'  # mastoides + ECG + Fp1 extra
        elif cu == 'STATUS':
            tipos[ch] = 'stim'

    if tipos:
        raw.set_channel_types(tipos)

    # Montaje BioSemi64 para coordenadas de electrodos
    try:
        montaje = mne.channels.make_standard_montage('biosemi64')
        raw.set_montage(montaje, on_missing='ignore', verbose=False)
    except Exception:
        pass

    return raw


# ══════════════════════════════════════════════════════════════
#  PASO 3 — RECORTAR: últimos 10 minutos
# ══════════════════════════════════════════════════════════════

def recortar_ultimos_10min(raw: mne.io.Raw) -> mne.io.Raw:
    """
    El paper de Sihn 2024 analiza los últimos 10 min de la sesión,
    asumiendo que el sujeto ya está inmerso en la tarea.
    Si la grabación es más corta, usa todo lo disponible.
    """
    duracion_total = raw.times[-1]

    if duracion_total <= DURACION_MIN:
        print(f"    Grabación corta ({duracion_total/60:.1f} min) — usando todo")
        return raw

    tmin = duracion_total - DURACION_MIN
    print(f"    Recortando: {tmin/60:.1f} min → {duracion_total/60:.1f} min")
    return raw.crop(tmin=tmin)


# ══════════════════════════════════════════════════════════════
#  PASO 4 — FILTRADO 0.5–45 Hz
# ══════════════════════════════════════════════════════════════

def filtrar(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Filtro FIR con ventana de Hamming, igual que en Sihn 2024.

    ¿Por qué 0.5 Hz abajo?
      Elimina la deriva DC lenta del BioSemi (grabado en DC coupling)
      sin destruir la actividad delta (>0.5 Hz). Si quisieras analizar
      ISA (<0.1 Hz) como en el paper, necesitarías NO filtrar aquí
      y trabajar la señal DC por separado.

    ¿Por qué 45 Hz arriba?
      Elimina el ruido de línea (50 Hz en Argentina) y artefactos
      musculares sin tocar gamma bajo (30-45 Hz).
    """
    print(f"    Filtrando {L_FREQ}–{H_FREQ} Hz (FIR Hamming)...")
    return raw.filter(
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        method='fir',
        fir_window='hamming',
        picks=['eeg', 'eog'],
        verbose=False
    )


# ══════════════════════════════════════════════════════════════
#  PASO 5 — RE-REFERENCIA AL PROMEDIO
# ══════════════════════════════════════════════════════════════

def rereferencia(raw: mne.io.Raw) -> mne.io.Raw:
    """
    El BioSemi graba con referencia al mastoides derecho (EXG6).
    Re-referenciamos al promedio de todos los electrodos EEG,
    que es la práctica estándar para análisis de potencia.

    IMPORTANTE: esto cambia la escala de amplitud de todos los canales.
    Es normal que después de esto la amplitud baje un poco.
    """
    print("    Re-referencia al promedio...")
    raw.set_eeg_reference('average', projection=False, verbose=False)
    return raw


# ══════════════════════════════════════════════════════════════
#  PASO 6 — DETECCIÓN DE CANALES MALOS
# ══════════════════════════════════════════════════════════════

def detectar_canales_malos(raw: mne.io.Raw) -> list[str]:
    """
    Estrategia simple basada en z-score de la varianza por canal.
    Un canal es "malo" si su varianza es muy diferente del resto.

    Umbral: UMBRAL_BAD_CH desviaciones estándar sobre la media.

    Para análisis más riguroso se puede usar el plugin RANSAC de MNE,
    pero requiere autoreject: pip install autoreject
    """
    picks_eeg = mne.pick_types(raw.info, eeg=True)
    data      = raw.get_data(picks=picks_eeg)

    # Varianza por canal
    var       = np.var(data, axis=1)
    z_scores  = (var - np.mean(var)) / np.std(var)

    malos = [raw.ch_names[picks_eeg[i]]
             for i, z in enumerate(z_scores)
             if abs(z) > UMBRAL_BAD_CH]

    if malos:
        print(f"    Canales malos detectados: {malos}")
    else:
        print("    No se detectaron canales malos")

    return malos


def interpolar_malos(raw: mne.io.Raw, malos: list[str]) -> mne.io.Raw:
    """
    Interpola canales malos usando interpolación esférica spline.
    Solo funciona si el montaje tiene coordenadas 3D (biosemi64 sí las tiene).
    """
    if not malos:
        return raw
    raw.info['bads'] = malos
    raw.interpolate_bads(reset_bads=True, verbose=False)
    print(f"    Interpolados: {malos}")
    return raw


# ══════════════════════════════════════════════════════════════
#  PASO 7 — DOWNSAMPLE
# ══════════════════════════════════════════════════════════════

def downsample(raw: mne.io.Raw) -> mne.io.Raw:
    """
    1024 Hz es excesivo para análisis de bandas hasta 45 Hz.
    Bajamos a 256 Hz: es suficiente (criterio Nyquist: 2 × 45 = 90 Hz mínimo)
    y reduce el tamaño de los datos ~4x.

    MNE aplica automáticamente un filtro anti-aliasing antes de resamplear.
    """
    if raw.info['sfreq'] > SFREQ_TARGET:
        print(f"    Downsample: {raw.info['sfreq']} Hz → {SFREQ_TARGET} Hz")
        raw = raw.resample(SFREQ_TARGET, verbose=False)
    return raw


# ══════════════════════════════════════════════════════════════
#  CONTROL DE CALIDAD — guardar figura PSD
# ══════════════════════════════════════════════════════════════

def guardar_psd_qc(raw: mne.io.Raw, sujeto: str, grupo: str):
    """
    Guarda una figura del espectro de potencia para inspección visual.
    Es el control de calidad más básico: si el espectro tiene picos
    extraños o mucho ruido muscular (>30 Hz muy elevado), algo salió mal.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))

    psd = raw.compute_psd(fmin=0.5, fmax=45, picks='eeg', verbose=False)
    psd_data  = psd.get_data()           # shape: (n_canales, n_freqs)
    freqs     = psd.freqs
    psd_media = np.mean(psd_data, axis=0)

    ax.semilogy(freqs, psd_media, color='steelblue', linewidth=1.2)

    # Sombrear bandas
    bandas = [
        (0.5, 4,  'delta',  'blue',   0.10),
        (4,   8,  'theta',  'cyan',   0.12),
        (8,  12,  'alpha',  'green',  0.15),
        (13, 30,  'beta',   'orange', 0.10),
        (30, 45,  'gamma',  'red',    0.08),
    ]
    for fmin, fmax, nombre, color, alpha in bandas:
        ax.axvspan(fmin, fmax, alpha=alpha, color=color, label=nombre)

    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('PSD (µV²/Hz)')
    ax.set_title(f'PSD post-preprocesamiento — {sujeto} [{grupo}]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ruta_fig = FIGURES_DIR / f"{sujeto}_psd_qc.png"
    fig.savefig(ruta_fig, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"    QC guardado: {ruta_fig}")


# ══════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL — un sujeto completo
# ══════════════════════════════════════════════════════════════

def preprocesar_sujeto(sujeto: str, grupo: str) -> mne.io.Raw | None:
    """
    Ejecuta el pipeline completo para un sujeto y guarda el resultado.
    Devuelve el objeto Raw limpio, o None si algo falló.
    """
    print(f"\n{'─'*55}")
    print(f"  {sujeto}  [{grupo}]")
    print(f"{'─'*55}")

    # ─ Paso 1: cargar
    raw = cargar_bdf(sujeto)
    if raw is None:
        return None

    # ─ Paso 2: tipos de canales
    raw = asignar_tipos(raw)

    # ─ Paso 3: recortar últimos 10 min
    raw = recortar_ultimos_10min(raw)

    # ─ Paso 4: filtrar
    raw = filtrar(raw)

    # ─ Paso 5: re-referencia
    raw = rereferencia(raw)

    # ─ Paso 6: canales malos
    malos = detectar_canales_malos(raw)
    raw   = interpolar_malos(raw, malos)

    # ─ Paso 7: downsample
    raw = downsample(raw)

    # ─ Control de calidad
    guardar_psd_qc(raw, sujeto, grupo)

    # ─ Guardar .fif
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ruta_out = OUTPUT_DIR / f"{sujeto}_{grupo}_prep-raw.fif"
    raw.save(str(ruta_out), overwrite=True, verbose=False)
    print(f"    Guardado: {ruta_out}")

    return raw


# ══════════════════════════════════════════════════════════════
#  MAIN — iterar sobre todos los sujetos
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*55)
    print("  PREPROCESAMIENTO EEG — Meditadores vs Controles")
    print("═"*55)

    resultados = {}  # {sujeto: raw}
    errores    = []

    for grupo, sujetos in GRUPOS.items():
        for sujeto in sujetos:
            try:
                raw = preprocesar_sujeto(sujeto, grupo)
                if raw is not None:
                    resultados[sujeto] = (grupo, raw)
            except Exception as e:
                print(f"  [ERROR] {sujeto}: {e}")
                errores.append(sujeto)

    # Resumen final
    print("\n" + "═"*55)
    print(f"  RESUMEN")
    print(f"{'─'*55}")
    print(f"  Procesados exitosamente : {len(resultados)}")
    print(f"  Con errores / no hallados: {len(errores)}")
    if errores:
        print(f"  Sujetos con error: {errores}")
    print(f"  Archivos guardados en  : {OUTPUT_DIR.resolve()}")
    print(f"  Figuras QC en          : {FIGURES_DIR.resolve()}")
    print("═"*55 + "\n")

    return resultados


if __name__ == '__main__':
    main()