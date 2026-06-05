"""
Módulo 3: hilbert / analytic.py
Responsabilidade: Cálculo do Sinal Analítico (Complexo) via Transformada de Hilbert (FFT).

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
from scipy.signal import hilbert

class AnalyticSignal:
    """
    Constrói o sinal analítico (z(t) = x(t) + i*y(t)) a partir das IMFs reais.
    Utiliza a transformada de Hilbert baseada em FFT (scipy.signal.hilbert).
    """

    @staticmethod
    def compute(imfs: np.ndarray) -> np.ndarray:
        """
        Calcula o sinal analítico.

        Parâmetros:
        -----------
        imfs : np.ndarray
            Sinal real ou matriz de IMFs (n_imfs, n_samples).

        Retorna:
        --------
        analytic_signal : np.ndarray
            Sinal complexo de mesmas dimensões.
        """
        # scipy.signal.hilbert calcula o sinal analítico ao longo do último eixo
        return hilbert(imfs, axis=-1)
