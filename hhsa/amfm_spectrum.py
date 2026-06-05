"""
Módulo 4: hhsa / amfm_spectrum.py
Responsabilidade: Estrutura de dados para armazenar e processar o resultado da HHSA.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
from .marginal import MarginalSpectrum

class AMFMSpectrum:
    """
    Objeto que armazena a matriz final do espectro (f_AM x f_FM)
    e fornece métodos rápidos para extração das características.
    """
    def __init__(self, spectrum_matrix: np.ndarray, fm_edges: np.ndarray, am_edges: np.ndarray):
        self.matrix = spectrum_matrix
        self.fm_edges = fm_edges
        self.am_edges = am_edges

    def get_marginal_fm(self) -> np.ndarray:
        """Integra a modulação AM para obter o espectro de Hilbert clássico (Marginal FM)."""
        return np.sum(self.matrix, axis=0)

    def get_marginal_am(self) -> np.ndarray:
        """Integra a portadora FM para visualizar a distribuição dominante de modulações AM."""
        return np.sum(self.matrix, axis=1)
