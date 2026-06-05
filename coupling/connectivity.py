"""
Módulo 5: coupling / connectivity.py
Responsabilidade: Matrizes de Conectividade Funcional para grafos neurais.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
from .coherence import HHSACoherence

class FunctionalConnectivity:
    """
    Constrói matrizes de adjacência (N_canais x N_canais) que servem de entrada
    para algoritmos de Teoria dos Grafos, Topologia de Mundo Pequeno e 
    Modelos de Decodificação Cerebral baseados em conectomas funcionais (FC).
    """

    @staticmethod
    def build_hhs_adjacency_matrix(hhs_matrices: list) -> np.ndarray:
        """
        Calcula a matriz de conectividade NxN par-a-par usando a coerência HHS.

        Parâmetros:
        -----------
        hhs_matrices : list of np.ndarray
            Lista contendo os espectros 2D (AMxFM) para cada um dos N canais.

        Retorna:
        --------
        adj_matrix : np.ndarray
            Matriz simétrica NxN com diagonal 1.0, representando a força
            da conexão funcional entre as diferentes ROIs.
        """
        n_channels = len(hhs_matrices)
        adj_matrix = np.ones((n_channels, n_channels))

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                coh_val = HHSACoherence.compute_coherence(hhs_matrices[i], hhs_matrices[j])
                adj_matrix[i, j] = coh_val
                adj_matrix[j, i] = coh_val

        return adj_matrix
