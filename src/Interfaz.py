'''
Visualizador de EEG con ipwidgets

Flujo:
Al ejecutar la celda, aparece un selector de sujeto (.bdf de ./data).
Al elegir el sujeto, se carga el archivo y aparecen pestañas:
    - EEG: señal cruda, navegable en ventanas de 30s (o 10/60s), por canal.
   - PSD: periodograma de Welch del canal completo seleccionado.
   - ISA: barras con la mediana del envelope ISA (0.03-0.08Hz) por región.
   - Aperiódico: ajuste specparam (componente aperiódica) por región.
   - Asimetría frontal: índice FAA (F4 vs F3, banda alfa).
'''
import sys
import glob
import re
from pathlib import Path

import mne
import numpy as np
from specparam import SpectralModel

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

mne.set_log_level('WARNING')

DATA_DIR = Path("./data")

REGIONES = {
    "frontal": ["Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
    "central": ["Cz", "C1", "C2", "C3", "C4", "C5", "C6"],
    "parietal": ["Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
    "occipital": ["Oz", "O1", "O2"],
}

BANDAS_PSD = [(1, 4, 'delta', 'blue'), (4, 8, 'theta', 'cyan'),
              (8, 13, 'alfa', 'green'), (13, 30, 'beta', 'orange')]


# ══════════════════════════════════════════════════════════════
#  Funciones de cómputo
# ══════════════════════════════════════════════════════════════
def calcular_isa_por_region(raw):
    resultado = {}
    for region, canales in REGIONES.items():
        canales_validos = [c for c in canales if c in raw.ch_names]
        if not canales_validos:
            continue
        raw_isa = raw.copy().pick(canales_validos)
        raw_isa.resample(sfreq=8, verbose=False)
        raw_isa.filter(l_freq=0.03, h_freq=0.08, method='fir',
                        fir_window='hamming', verbose=False)
        raw_isa.apply_hilbert(envelope=True)
        data = raw_isa.get_data()
        mediana_por_canal = np.median(data, axis=1)
        resultado[region] = np.mean(mediana_por_canal)
    return resultado


def calcular_aperiodico_por_region(raw):
    resultado = {}
    canales_todos = [c for c in raw.ch_names
                      if any(c in canales for canales in REGIONES.values())]
    raw_filtrado = raw.copy().pick(canales_todos)
    raw_filtrado.filter(l_freq=0.5, h_freq=35, verbose=False)

    for region, canales in REGIONES.items():
        canales_validos = [c for c in canales if c in raw_filtrado.ch_names]
        if not canales_validos:
            continue
        exponentes = []
        for canal in canales_validos:
            data = raw_filtrado.get_data(picks=canal)[0]
            psds_f, freqs_f = mne.time_frequency.psd_array_welch(
                data, sfreq=raw.info['sfreq'], n_fft=4096, fmax=100,
                n_per_seg=4096, output="power", verbose=False
            )
            mask = (freqs_f >= 0.5) & (freqs_f <= 50)
            try:
                fm = SpectralModel(min_peak_height=0.1, verbose=False)
                fm.fit(freqs_f[mask], psds_f[mask])
                exponentes.append(fm.results.params.aperiodic.params[1])
            except Exception:
                continue
        if exponentes:
            resultado[region] = np.mean(exponentes)
    return resultado


def calcular_asimetria_frontal(raw):
    if "F3" not in raw.ch_names or "F4" not in raw.ch_names:
        return None

    raw_filtrado = raw.copy().pick(["F3", "F4"])
    raw_filtrado.filter(l_freq=0.5, h_freq=35, verbose=False)

    potencias = {}
    for canal in ["F3", "F4"]:
        data = raw_filtrado.get_data(picks=canal)[0]
        psds_f, freqs_f = mne.time_frequency.psd_array_welch(
            data, sfreq=raw.info['sfreq'], n_fft=4096, fmax=100,
            n_per_seg=4096, output="power", verbose=False
        )
        mask = (freqs_f >= 8) & (freqs_f <= 13)
        potencias[canal] = np.trapezoid(psds_f[mask], freqs_f[mask])

    return np.log(potencias["F4"]) - np.log(potencias["F3"])


# ══════════════════════════════════════════════════════════════
#  Canvas genérico de matplotlib embebido
# ══════════════════════════════════════════════════════════════
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=(7, 4)):
        self.fig = Figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def limpiar(self):
        self.ax.clear()


# ══════════════════════════════════════════════════════════════
#  Pestaña EEG
# ══════════════════════════════════════════════════════════════
class TabEEG(QtWidgets.QWidget):
    def __init__(self, raw, parent=None):
        super().__init__(parent)
        self.raw_original = raw
        self.raw_proc = None
        self.usar_proc = False
        self.raw = raw
        self.dur_total = raw.times[-1]
        self.canales_eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]

        layout = QtWidgets.QVBoxLayout(self)

        # controles
        controles = QtWidgets.QHBoxLayout()
        self.canal_combo = QtWidgets.QComboBox()
        self.canal_combo.addItems(self.canales_eeg)
        if "F3" in self.canales_eeg:
            self.canal_combo.setCurrentText("F3")

        self.dur_combo = QtWidgets.QComboBox()
        self.dur_combo.addItems(["2 s", "5 s", "10 s", "30 s", "60 s"])
        self.dur_combo.setCurrentText("30 s")

        self.boton_prep = QtWidgets.QPushButton("Aplicar preprocesamiento")

        controles.addWidget(QtWidgets.QLabel("Canal:"))
        controles.addWidget(self.canal_combo)
        controles.addWidget(QtWidgets.QLabel("Ventana:"))
        controles.addWidget(self.dur_combo)
        controles.addStretch()
        controles.addWidget(self.boton_prep)
        layout.addLayout(controles)

        # slider de tiempo
        slider_layout = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.label_tiempo = QtWidgets.QLabel("0 s")
        slider_layout.addWidget(QtWidgets.QLabel("Tiempo:"))
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.label_tiempo)
        layout.addLayout(slider_layout)

        # canvas
        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas)

        # conexiones
        self.canal_combo.currentTextChanged.connect(self.graficar)
        self.dur_combo.currentTextChanged.connect(self.actualizar_slider)
        self.slider.valueChanged.connect(self.graficar)
        self.boton_prep.clicked.connect(self.toggle_preprocesamiento)

        self.actualizar_slider()

    def toggle_preprocesamiento(self):
        if self.raw_proc is None:
            self.boton_prep.setEnabled(False)
            self.boton_prep.setText("Procesando...")
            QtWidgets.QApplication.processEvents()
            proc = self.raw_original.copy()
            proc.filter(l_freq=0.5, h_freq=35, method='fir', fir_window='hamming', verbose=False)
            proc.notch_filter(freqs=16, verbose=False)
            self.raw_proc = proc
            self.boton_prep.setEnabled(True)

        self.usar_proc = not self.usar_proc
        self.raw = self.raw_proc if self.usar_proc else self.raw_original
        self.boton_prep.setText(
            "Quitar preprocesamiento" if self.usar_proc else "Aplicar preprocesamiento"
        )
        self.graficar()

    def duracion(self):
        return int(self.dur_combo.currentText().split()[0])

    def actualizar_slider(self):
        dur = self.duracion()
        max_t = max(0, int(self.dur_total - dur))
        self.slider.setMaximum(max_t)
        self.slider.setValue(0)
        self.graficar()

    def graficar(self):
        canal = self.canal_combo.currentText()
        dur = self.duracion()
        t0 = self.slider.value()
        t1 = min(t0 + dur, self.dur_total)
        self.label_tiempo.setText(f"{t0:.0f} s")

        data, times = self.raw.copy().pick([canal]).get_data(tmin=t0, tmax=t1, return_times=True)
        senal = data[0] * 1e6  # µV
        senal -= senal.mean()  # eliminar offset DC para centrar la señal

        self.canvas.limpiar()
        ax = self.canvas.ax
        ax.plot(times, senal, color='steelblue', linewidth=0.7)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud (µV)")
        ax.set_title(f"EEG — canal {canal} | ventana [{t0:.0f}-{t1:.0f}] s")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-125, 125)
        self.canvas.fig.tight_layout()
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════
#  Pestaña PSD
# ══════════════════════════════════════════════════════════════
class TabPSD(QtWidgets.QWidget):
    def __init__(self, raw, parent=None):
        super().__init__(parent)
        self.raw_original = raw
        self.raw_proc = None
        self.usar_proc = False
        self.raw = raw
        self.canales_eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]

        layout = QtWidgets.QVBoxLayout(self)

        controles = QtWidgets.QHBoxLayout()
        self.canal_combo = QtWidgets.QComboBox()
        self.canal_combo.addItems(self.canales_eeg)
        if "F3" in self.canales_eeg:
            self.canal_combo.setCurrentText("F3")

        self.boton_prep = QtWidgets.QPushButton("Aplicar preprocesamiento")

        controles.addWidget(QtWidgets.QLabel("Canal:"))
        controles.addWidget(self.canal_combo)
        controles.addStretch()
        controles.addWidget(self.boton_prep)
        layout.addLayout(controles)

        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas)

        self.canal_combo.currentTextChanged.connect(self.graficar)
        self.boton_prep.clicked.connect(self.toggle_preprocesamiento)

        self.graficar()

    def toggle_preprocesamiento(self):
        if self.raw_proc is None:
            self.boton_prep.setEnabled(False)
            self.boton_prep.setText("Procesando...")
            QtWidgets.QApplication.processEvents()
            proc = self.raw_original.copy()
            proc.filter(l_freq=0.5, h_freq=35, method='fir', fir_window='hamming', verbose=False)
            proc.notch_filter(freqs=16, verbose=False)
            self.raw_proc = proc
            self.boton_prep.setEnabled(True)

        self.usar_proc = not self.usar_proc
        self.raw = self.raw_proc if self.usar_proc else self.raw_original
        self.boton_prep.setText(
            "Quitar preprocesamiento" if self.usar_proc else "Aplicar preprocesamiento"
        )
        self.graficar()

    def graficar(self):
        canal = self.canal_combo.currentText()
        senal = self.raw.get_data(picks=canal)[0]

        n_fft = min(4096, len(senal))
        psds, freqs = mne.time_frequency.psd_array_welch(
            senal, sfreq=self.raw.info['sfreq'], fmax=50,
            n_fft=n_fft, n_per_seg=n_fft, output="power", verbose=False
        )

        self.canvas.limpiar()
        ax = self.canvas.ax
        ax.semilogy(freqs, psds, color='darkorange')
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("PSD (V²/Hz)")
        ax.set_title(f"PSD — canal {canal} | registro completo")
        ax.grid(True, alpha=0.3)
        for fmin, fmax, nombre, color in BANDAS_PSD:
            ax.axvspan(fmin, fmax, alpha=0.1, color=color, label=nombre)
        ax.legend(fontsize=8)
        self.canvas.fig.tight_layout()
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════
#  Pestaña ISA
# ══════════════════════════════════════════════════════════════
class TabISA(QtWidgets.QWidget):
    def __init__(self, raw, parent=None):
        super().__init__(parent)
        self.raw = raw

        layout = QtWidgets.QVBoxLayout(self)
        self.boton = QtWidgets.QPushButton("Calcular ISA por región")
        self.canvas = MplCanvas(self)
        layout.addWidget(self.boton)
        layout.addWidget(self.canvas)

        self.boton.clicked.connect(self.calcular)

    def calcular(self):
        self.boton.setText("Calculando...")
        QtWidgets.QApplication.processEvents()

        resultado = calcular_isa_por_region(self.raw)

        self.canvas.limpiar()
        ax = self.canvas.ax
        ax.bar(list(resultado.keys()), list(resultado.values()), color='mediumseagreen')
        ax.set_ylabel("Mediana del envelope ISA (a.u.)")
        ax.set_title("ISA (0.03-0.08 Hz) por región")
        ax.grid(True, alpha=0.3, axis='y')
        self.canvas.fig.tight_layout()
        self.canvas.draw()

        self.boton.setText("Calcular ISA por región")


# ══════════════════════════════════════════════════════════════
#  Pestaña Aperiódico
# ══════════════════════════════════════════════════════════════
class TabAperiodico(QtWidgets.QWidget):
    def __init__(self, raw, parent=None):
        super().__init__(parent)
        self.raw = raw

        layout = QtWidgets.QVBoxLayout(self)
        self.boton = QtWidgets.QPushButton("Calcular componente aperiódica")
        self.canvas = MplCanvas(self)
        layout.addWidget(self.boton)
        layout.addWidget(self.canvas)

        self.boton.clicked.connect(self.calcular)

    def calcular(self):
        self.boton.setText("Calculando (puede tardar)...")
        QtWidgets.QApplication.processEvents()

        resultado = calcular_aperiodico_por_region(self.raw)

        self.canvas.limpiar()
        ax = self.canvas.ax
        ax.bar(list(resultado.keys()), list(resultado.values()), color='indianred')
        ax.set_ylabel("Exponente aperiódico")
        ax.set_title("Componente aperiódica (specparam) por región")
        ax.grid(True, alpha=0.3, axis='y')
        self.canvas.fig.tight_layout()
        self.canvas.draw()

        self.boton.setText("Calcular componente aperiódica")


# ══════════════════════════════════════════════════════════════
#  Pestaña Asimetría frontal
# ══════════════════════════════════════════════════════════════
class TabAsimetria(QtWidgets.QWidget):
    def __init__(self, raw, parent=None):
        super().__init__(parent)
        self.raw = raw

        layout = QtWidgets.QVBoxLayout(self)
        self.boton = QtWidgets.QPushButton("Calcular asimetría frontal (F4 - F3)")
        self.canvas = MplCanvas(self, figsize=(5, 4))
        self.label_info = QtWidgets.QLabel("")
        layout.addWidget(self.boton)
        layout.addWidget(self.canvas)
        layout.addWidget(self.label_info)

        self.boton.clicked.connect(self.calcular)

    def calcular(self):
        asimetria = calcular_asimetria_frontal(self.raw)

        if asimetria is None:
            self.label_info.setText("No se encontraron los canales F3/F4 en este sujeto.")
            return

        self.canvas.limpiar()
        ax = self.canvas.ax
        color = 'royalblue' if asimetria >= 0 else 'firebrick'
        ax.bar(["FAA (F4 - F3)"], [asimetria], color=color)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel("log(potencia F4) - log(potencia F3)")
        ax.set_title("Índice de asimetría frontal alfa")
        ax.grid(True, alpha=0.3, axis='y')
        self.canvas.fig.tight_layout()
        self.canvas.draw()

        signo = "F4 (hemisferio derecho)" if asimetria >= 0 else "F3 (hemisferio izquierdo)"
        self.label_info.setText(f"Valor: {asimetria:.4f}  →  mayor potencia alfa en {signo}")


# ══════════════════════════════════════════════════════════════
#  Widget principal del sujeto (con pestañas)
# ══════════════════════════════════════════════════════════════
class MyWidget(QtWidgets.QWidget):
    def __init__(self, raw, sujeto_label, parent=None):
        super().__init__(parent)
        self.raw = raw

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"<b>Sujeto: {sujeto_label}</b>"))

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(TabEEG(raw), "EEG")
        tabs.addTab(TabPSD(raw), "PSD")
        tabs.addTab(TabISA(raw), "ISA")
        tabs.addTab(TabAperiodico(raw), "Aperiódico")
        tabs.addTab(TabAsimetria(raw), "Asimetría frontal")

        layout.addWidget(tabs)


# ══════════════════════════════════════════════════════════════
#  Ventana principal: selector de sujeto
# ══════════════════════════════════════════════════════════════
class VentanaPrincipal(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualizador EEG — TP PSIB")
        self.resize(1000, 700)

        self.opciones_sujetos = self._listar_sujetos()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        self.layout = QtWidgets.QVBoxLayout(central)

        selector_layout = QtWidgets.QHBoxLayout()
        self.combo_sujeto = QtWidgets.QComboBox()
        self.combo_sujeto.addItems(list(self.opciones_sujetos.keys()))
        self.boton_abrir = QtWidgets.QPushButton("Abrir sujeto")
        selector_layout.addWidget(QtWidgets.QLabel("Sujeto:"))
        selector_layout.addWidget(self.combo_sujeto)
        selector_layout.addWidget(self.boton_abrir)
        self.layout.addLayout(selector_layout)

        self.contenedor = QtWidgets.QWidget()
        self.layout.addWidget(self.contenedor)

        self.boton_abrir.clicked.connect(self.abrir_sujeto)

    def _listar_sujetos(self):
        archivos_bdf = sorted(glob.glob(str(DATA_DIR / "*.bdf")))
        opciones = {}
        for f in archivos_bdf:
            m = re.search(r'sub-(\d+)', f)
            if m:
                numero = int(m.group(1))
                grupo = "CTR" if numero < 60 else "EXP"
                label = f"sub-{numero:03d} [{grupo}]"
                opciones[label] = f
        if not opciones:
            raise RuntimeError(f"No se encontraron archivos .bdf en {DATA_DIR.resolve()}")
        return opciones

    def abrir_sujeto(self):
        label = self.combo_sujeto.currentText()
        archivo = self.opciones_sujetos[label]

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            raw = mne.io.read_raw_bdf(archivo, preload=True, verbose=False)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        nuevo = MyWidget(raw, label)

        # reemplazar el contenedor actual
        self.layout.removeWidget(self.contenedor)
        self.contenedor.deleteLater()
        self.contenedor = nuevo
        self.layout.addWidget(self.contenedor)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec_())