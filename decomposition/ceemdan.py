"""
Módulo 2: decomposition / ceemdan.py
Responsabilidade: Complete Ensemble Empirical Mode Decomposition with Adaptive Noise.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
import warnings

try:
    from PyEMD import CEEMDAN
except ImportError:
    warnings.warn("PyEMD não encontrado. Instale com: pip install EMD-signal")
    CEEMDAN = None

class CEEMDANDecomposer:
    """
    Wrapper para CEEMDAN (Torres et al., 2011).
    Adiciona ruído branco em cada estágio da decomposição e calcula o resíduo médio
    para garantir reconstrução exata do sinal e mitigar mode-mixing.
    """

    def __init__(self, trials: int = 100, noise_scale: float = 0.2, max_imfs: int = -1):
        self.trials = trials
        self.noise_scale = noise_scale
        self.max_imfs = max_imfs
        if CEEMDAN is not None:
            self.instance = CEEMDAN(trials=self.trials, epsilon=self.noise_scale, ext_EMD=None)
        else:
            self.instance = None

    def decompose(self, signal: np.ndarray, time_axis: np.ndarray = None) -> np.ndarray:
        if self.instance is None:
            raise ImportError("Biblioteca PyEMD não está disponível.")

        if time_axis is None:
            time_axis = np.arange(len(signal))

        imfs = self.instance.ceemdan(signal, T=time_axis, max_imf=self.max_imfs)
        return imfs
