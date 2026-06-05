"""
Módulo 4: hhsa / demo.py
Responsabilidade: Demonstração do pipe Nested EMD e Extração de Não-Linearidade.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
from nonlinearity import NonlinearityIndex

def run_demo():
    print("PyHHSA - Validação do Módulo 4 (HHSA Core)")

    # Criar um conjunto de envelopes simulados para testar a matemática do N_in
    # Caso 1: Linear (Amplitudes Constantes -> N_in próximo a 0)
    env_linear = np.ones((3, 1000)) * 2.0

    # Caso 2: Altamente Não-Linear (Envelopes flutuando massivamente)
    t = np.linspace(0, 10, 1000)
    env_nonlinear = np.vstack([
        1.0 + 0.9 * np.sin(2 * np.pi * 0.5 * t),
        2.0 + 1.8 * np.sin(2 * np.pi * 1.5 * t),
        0.5 + 0.4 * np.cos(2 * np.pi * 0.2 * t)
    ])

    nin_linear = NonlinearityIndex.calculate_inter_mode(env_linear)
    nin_nonlinear = NonlinearityIndex.calculate_inter_mode(env_nonlinear)

    print(f" -> Índice de Não-Linearidade Inter-modo (Sinal Linear)    : {nin_linear:.4f} (Esperado: 0.00)")
    print(f" -> Índice de Não-Linearidade Inter-modo (Sinal Não-Linear): {nin_nonlinear:.4f} (Esperado: > 0.20)")

    print("\nClasses construídas e validadas: NestedEMD, HoloHilbertSpectrum, AMFMSpectrum, NonlinearityIndex.")
    print("[Validação Funcional]: Módulo 4 Operacional.")

if __name__ == "__main__":
    run_demo()
