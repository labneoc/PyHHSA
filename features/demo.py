"""
Módulo 7: features / demo.py
Responsabilidade: Validação da extração e da formatação para Machine Learning.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
from roi_power import ROIPower
from mean_freq import MeanFrequency
from feature_matrix import FeatureMatrixBuilder

def run_demo():
    print("PyHHSA - Validação do Módulo 7 (Features for Brain Decoding)")

    # Simula uma matriz HHS (AM x FM) extraída de um córtex sensório-motor
    am_bins, fm_bins = 20, 50
    spectrum = np.random.rand(am_bins, fm_bins)

    am_edges = np.linspace(0, 10, am_bins + 1)
    fm_edges = np.linspace(0, 50, fm_bins + 1)

    # 1. ROIPower
    # Extrair P_total na banda Beta (13-30 Hz) modulada por Delta AM (0-4 Hz)
    beta_delta_power = ROIPower.extract_power_in_band(spectrum, fm_edges, am_edges, 
                                                      fm_band=(13, 30), am_band=(0, 4))
    print(f" -> Extração de P_Total (Beta portadora, Delta moduladora): {beta_delta_power:.2f}")

    # 2. Mean Frequencies
    cf_fm = MeanFrequency.calculate_cf_fm(spectrum, fm_edges)
    cf_am = MeanFrequency.calculate_cf_am(spectrum, am_edges)
    print(f" -> Frequência Média Portadora (cf_FM): {cf_fm:.2f} Hz")
    print(f" -> Frequência Média Moduladora (cf_AM): {cf_am:.2f} Hz")

    # 3. Builder Scikit-Learn
    try:
        builder = FeatureMatrixBuilder()
        builder.add_trial(subject_id='S01', label=1, features={'C3_P_Beta': 12.4, 'C3_cf_FM': 22.1})
        builder.add_trial(subject_id='S01', label=0, features={'C3_P_Beta':  9.2, 'C3_cf_FM': 18.5})

        X, y, names = builder.export_sklearn_format()
        print("\n -> Feature Matrix X (Shape):", X.shape)
        print(" -> Rótulos y (Classes):", y)
        print(" -> Colunas:", names)
        print("\n[Validação Funcional]: Módulo 7 Operacional. Pronto para o SVM.")
    except Exception as e:
        print(f"Erro no Feature Builder (Pandas ausente?): {e}")

if __name__ == "__main__":
    run_demo()
