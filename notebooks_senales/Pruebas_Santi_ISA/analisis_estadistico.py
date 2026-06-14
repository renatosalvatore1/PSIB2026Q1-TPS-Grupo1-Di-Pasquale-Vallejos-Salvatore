"""
Lee los archivos *_graphpad.xlsx generados por exportar_resultado.py
y aplica el test de Wilcoxon rank-sum (Mann-Whitney U) unilateral
(EXP < CTR), con corrección FDR (Benjamini-Hochberg) cuando hay
múltiples comparaciones (varias hojas / canales).

Requiere: pandas, scipy, statsmodels, openpyxl
    pip install scipy statsmodels --break-system-packages
"""

import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def analizar_archivo(path_excel, nombre_valor="valor", alternative="less"):
    """
    Recorre todas las hojas de un archivo Excel con columnas EXP y CTR,
    aplica Mann-Whitney unilateral por hoja y corrige por FDR.

    alternative='less'  -> hipótesis: EXP < CTR (esperado en ISA, según el paper)
    alternative='greater' -> hipótesis: EXP > CTR
    """
    xls = pd.ExcelFile(path_excel)
    resultados = []

    for hoja in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=hoja)

        # columnas EXP / CTR, sin NaN
        exp = df["EXP"].dropna().values
        ctr = df["CTR"].dropna().values

        if len(exp) == 0 or len(ctr) == 0:
            print(f"  [SKIP] Hoja '{hoja}': datos insuficientes (EXP={len(exp)}, CTR={len(ctr)})")
            continue

        stat, p = mannwhitneyu(exp, ctr, alternative=alternative)

        resultados.append({
            "hoja": hoja,
            "n_exp": len(exp),
            "n_ctr": len(ctr),
            "U_stat": stat,
            "p_valor": p,
        })

    df_res = pd.DataFrame(resultados)

    if df_res.empty:
        print(f"  No se pudieron calcular resultados para {path_excel}")
        return df_res

    # Corrección FDR (Benjamini-Hochberg)
    rechazado, p_corregido, _, _ = multipletests(
        df_res["p_valor"].values, alpha=0.05, method="fdr_bh"
    )
    df_res["p_fdr"] = p_corregido
    df_res["significativo_fdr_0.05"] = rechazado

    return df_res


def main():
    archivos = {
        "ISA": ("ISA_graphpad.xlsx", "less"),
        "PSD": ("PSD_graphpad.xlsx", "less"),
        "Aperiodico": ("aperiodico_graphpad.xlsx", "less"),
        "Asimetria": ("asimetria_graphpad.xlsx", "two-sided"),
    }

    with pd.ExcelWriter("resultados_estadisticos.xlsx") as writer:
        for nombre, (archivo, alternativa) in archivos.items():
            print(f"\n=== {nombre} ({archivo}) ===")
            try:
                df_res = analizar_archivo(archivo, alternative=alternativa)
            except FileNotFoundError:
                print(f"  [ERROR] No se encontró {archivo}")
                continue

            if not df_res.empty:
                print(df_res.to_string(index=False))
                df_res.to_excel(writer, sheet_name=nombre, index=False)

    print("\nResultados guardados en: resultados_estadisticos.xlsx")


if __name__ == "__main__":
    main()
'''

Antes de cambio con # promedio por sujeto sobre los canales de la región/banda
            df_sujeto = df_sub.groupby(["sujeto", "grupo"])["potencia"].mean().reset_index()
en cada exportacion:
=== ISA (ISA_graphpad.xlsx) ===
     hoja  n_exp  n_ctr  U_stat      p_valor    p_fdr  significativo_fdr_0.05
  frontal     99     99  2940.0 5.820147e-07 0.000002                    True
  central     77     77  2403.0 2.132091e-02 0.028428                    True
 parietal     99     99  4003.0 1.304211e-02 0.026084                    True
occipital     33     33   404.0 3.629299e-02 0.036293                    True

=== PSD (PSD_graphpad.xlsx) ===
           hoja  n_exp  n_ctr  U_stat  p_valor    p_fdr  significativo_fdr_0.05
  frontal_delta     99     99  3483.0 0.000217 0.003474                    True
  frontal_theta     99     99  5022.0 0.619026 0.986004                   False
   frontal_alfa     99     99  6642.0 0.999992 0.999997                   False
   frontal_beta     99     99  3645.0 0.000916 0.007330                    True
  central_delta     77     77  2450.0 0.031496 0.100787                   False
  central_theta     77     77  3136.0 0.733082 0.986004                   False
   central_alfa     77     77  4018.0 0.999931 0.999997                   False
   central_beta     77     77  2156.0 0.001735 0.009253                    True
 parietal_delta     99     99  4779.0 0.381921 0.763842                   False
 parietal_theta     99     99  5103.0 0.692885 0.986004                   False
  parietal_alfa     99     99  6723.0 0.999997 0.999997                   False
  parietal_beta     99     99  4698.0 0.307988 0.703973                   False
occipital_delta     33     33   387.0 0.021937 0.087748                   False
occipital_theta     33     33   594.0 0.739503 0.986004                   False
 occipital_alfa     33     33   738.0 0.993617 0.999997                   False
 occipital_beta     33     33   468.0 0.164642 0.439046                   False

=== Aperiodico (aperiodico_graphpad.xlsx) ===
     hoja  n_exp  n_ctr  U_stat      p_valor    p_fdr  significativo_fdr_0.05
  frontal    396    396 89424.0 9.996934e-01 0.999693                   False
  central    308    308 40768.0 1.262675e-03 0.001684                    True
 parietal    396    396 64800.0 1.161112e-05 0.000023                    True
occipital    132    132  5616.0 2.935646e-07 0.000001                    True

=== Asimetria (asimetria_graphpad.xlsx) ===
hoja  n_exp  n_ctr  U_stat  p_valor  p_fdr  significativo_fdr_0.05
 FAA     11     11    60.5      1.0    1.0                   False

Resultados guardados en: resultados_estadisticos.xlsx


'''

'''

Despues de cambio con # promedio por sujeto sobre los canales de la región/banda
            df_sujeto = df_sub.groupby(["sujeto", "grupo"])["potencia"].mean().reset_index()
en cada exportacion:

=== ISA (ISA_graphpad.xlsx) ===
     hoja  n_exp  n_ctr  U_stat  p_valor    p_fdr  significativo_fdr_0.05
  frontal     11     11    30.0 0.024422 0.097688                   False
  central     11     11    48.0 0.215354 0.215354                   False
 parietal     11     11    48.0 0.215354 0.215354                   False
occipital     11     11    43.0 0.132146 0.215354                   False

=== PSD (PSD_graphpad.xlsx) ===
           hoja  n_exp  n_ctr  U_stat  p_valor    p_fdr  significativo_fdr_0.05
  frontal_delta     11     11    43.0 0.132146 0.649273                   False
  frontal_theta     11     11    62.0 0.552243 0.870941                   False
   frontal_alfa     11     11    82.0 0.925719 0.934516                   False
   frontal_beta     11     11    45.0 0.162318 0.649273                   False
  central_delta     11     11    50.0 0.255703 0.799148                   False
  central_theta     11     11    64.0 0.603594 0.870941                   False
   central_alfa     11     11    82.0 0.925719 0.934516                   False
   central_beta     11     11    44.0 0.146712 0.649273                   False
 parietal_delta     11     11    59.0 0.473822 0.870941                   False
 parietal_theta     11     11    63.0 0.578084 0.870941                   False
  parietal_alfa     11     11    83.0 0.934516 0.934516                   False
  parietal_beta     11     11    58.0 0.447757 0.870941                   False
occipital_delta     11     11    43.0 0.132146 0.649273                   False
occipital_theta     11     11    66.0 0.653206 0.870941                   False
 occipital_alfa     11     11    82.0 0.925719 0.934516                   False
 occipital_beta     11     11    52.0 0.299680 0.799148                   False

=== Aperiodico (aperiodico_graphpad.xlsx) ===
     hoja  n_exp  n_ctr  U_stat  p_valor    p_fdr  significativo_fdr_0.05
  frontal     11     11    69.0 0.722735 0.722735                   False
  central     11     11    52.0 0.299680 0.399574                   False
 parietal     11     11    50.0 0.255703 0.399574                   False
occipital     11     11    39.0 0.083953 0.335811                   False

=== Asimetria (asimetria_graphpad.xlsx) ===
hoja  n_exp  n_ctr  U_stat  p_valor  p_fdr  significativo_fdr_0.05
 FAA     11     11    60.5      1.0    1.0                   False

Resultados guardados en: resultados_estadisticos.xlsx

'''

'''

hay una tendencia (no una certeza estadística) de que el grupo EXP (meditadores) tiene menor amplitud de actividad infra-lenta (ISA) que el grupo CTR (controles), y esa tendencia es más marcada en la región frontal (p=0.024 sin corregir, p_fdr=0.098 con corrección).
En las otras tres regiones (central, parietal, occipital) los p-valores son más altos (0.13-0.22), o sea casi no hay diferencia entre grupos ahí.
Esto coincide con la dirección que encuentra el paper de Sihn et al. (2024): ellos también ven la reducción de ISA más fuerte en frontal en meditadores Vipassana. La diferencia es que su efecto sí logra ser estadísticamente significativo porque tienen más sujetos (19 EXP vs 31 CTR aprox.) que ustedes (11 vs 11).
Conclusión: no pueden afirmar "el ISA frontal es menor en meditadores" como hecho probado, pero sí pueden decir "observamos una tendencia consistente con la literatura, no significativa tras FDR, probablemente por tamaño de muestra".


PSD por bandas
Todos los p-valores son altos (0.13 a 0.93), ninguno cerca de ser significativo. Entonces que no hay evidencia de diferencias en la potencia espectral por bandas clásicas entre meditadores y controles, en ninguna región ni banda, con esta muestra.
Esto también es consistente con el paper: ellos encuentran el efecto específicamente en ISA (<0.1 Hz), no en las bandas EEG tradicionales (delta/theta/alfa/beta). Entonces este resultado "negativo" en PSD en realidad respalda la idea de que el efecto de la meditación es específico de la banda infra-lenta, y no se ve en las oscilaciones más rápidas.


Aperiodico
Tampoco hay nada significativo (p entre 0.08 y 0.72). Ya  que la componente aperiódica del espectro (relacionada con la "forma de fondo" del espectro de potencia, sin los picos de oscilación) no difiere entre grupos en ninguna región. Occipital es la más cercana (p=0.084) pero lejos de ser concluyente.

Asimetría frontal alfa (FAA)
p=1.0, totalmente sin diferencia. No hay asimetría diferencial entre F3 y F4 según el grupo — el índice de asimetría frontal alfa (relacionado con estados emocionales/motivacionales en otra literatura) no distingue meditadores de controles en este dataset.

POR ENDE
Unico resultado con algo de interés es la tendencia en ISA frontal, en la dirección esperada por la hipótesis del paper. 
El resto de las medidas (PSD por bandas, componente aperiódica, asimetría) no muestran diferencias, lo cual es consistente con que el paper original 
también encuentra el efecto de la meditación concentrado específicamente en la banda ISA (0.03-0.08 Hz) y no en otras métricas espectrales. La principal limitación es 
el tamaño muestral (n=11 por grupo vs n=19-31 del estudio original), que reduce la potencia para detectar efectos pequeños tras corrección por comparaciones múltiples.
'''