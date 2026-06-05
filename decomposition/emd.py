"""
Módulo 2: decomposition / emd.py
Responsabilidade: Implementação do algoritmo EMD clássico (Empirical Mode Decomposition).

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
import warnings

try:
    from PyEMD import EMD
except ImportError:
    warnings.warn("PyEMD não encontrado. Instale com: pip install EMD-signal")
    EMD = None

class EMDDecomposer:
    """Wrapper para a Decomposição Empírica de Modos (EMD) clássica."""

    def __init__(self, max_imfs: int = -1, spline_kind: str = 'cubic'):
        self.max_imfs = max_imfs
        self.spline_kind = spline_kind
        if EMD is not None:
            self.emd_instance = EMD(spline_kind=self.spline_kind)
            self.emd_instance.FIXE = 100  # Máximo de iterações de sifting para evitar loops infinitos
        else:
            self.emd_instance = None

    def decompose(self, signal: np.ndarray, time_axis: np.ndarray = None) -> np.ndarray:
        """
        Decompõe o sinal em Funções de Modo Intrínseco (IMFs).

        Parâmetros:
        -----------
        signal : np.ndarray
            Sinal 1D de entrada (ex: série temporal de um eletrodo).
        time_axis : np.ndarray, opcional
            Vetor de tempo.

        Retorna:
        --------
        imfs : np.ndarray
            Matriz 2D (n_imfs, n_samples).
        """
        if self.emd_instance is None:
            raise ImportError("Biblioteca PyEMD não está disponível.")

        if time_axis is None:
            time_axis = np.arange(len(signal))

        imfs = self.emd_instance.emd(signal, T=time_axis, max_imf=self.max_imfs)
        return imfs
