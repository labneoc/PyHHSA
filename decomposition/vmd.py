"""
Módulo 2: decomposition / vmd.py
Responsabilidade: Variational Mode Decomposition (VMD).

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
import warnings

try:
    from vmdpy import VMD
except ImportError:
    warnings.warn("vmdpy não encontrado. Instale com: pip install vmdpy")
    VMD = None

class VMDDecomposer:
    """
    Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014).
    Abordagem rigorosa baseada em minimização funcional (otimização variacional) 
    em vez de sifting recursivo. Sujeito a hiperparâmetros mais rígidos que a EMD.
    """

    def __init__(self, alpha: int = 2000, tau: float = 0.0, K: int = 5, DC: int = 0, init: int = 1, tol: float = 1e-7):
        self.alpha = alpha  # Penalidade de banda de frequência
        self.tau = tau      # Passo dual (Lagrange)
        self.K = K          # Número rígido de modos a extrair (requer conhecimento prévio)
        self.DC = DC        # Permitir componente contínuo (0 = False, 1 = True)
        self.init = init    # Inicialização das frequências ômega (1 = uniformemente distribuídas)
        self.tol = tol      # Tolerância para critério de parada convergente

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        if VMD is None:
            raise ImportError("Biblioteca vmdpy não está disponível.")

        u, u_hat, omega = VMD(signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
        # u contém as IMFs (K modos temporais)
        return u
