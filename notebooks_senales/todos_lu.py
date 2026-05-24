from pathlib import Path
import mne

# cargar el archivo .bdf
DATA_DIR = Path(__file__).parent.parent / 'data' #path relativo a la carpeta 'data' del repo (funciona en cualquier máquina)
bdf_files = sorted(DATA_DIR.glob('*.bdf')) #arma lista con todos los archivos .bdf

#trabaja con todos los archivos
for bdf_path in bdf_files:
    subject_id = bdf_path.stem
    print(f'Procesando: {subject_id}')
    raw = mne.io.read_raw_bdf(bdf_path, preload=True)
    
    #lo que le querramos hacer a todos los archivos

