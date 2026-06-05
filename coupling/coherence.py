"""
Módulo 5: coupling / coherence.py
Responsabilidade: Coerência Espectral Cross-Scale (HHSA Coherence).

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np

class HHSACoherence:
    """
    Estima a coerência baseada nos espectros de Holo-Hilbert.
    Permite avaliar se dois canais (ex: Córtex Motor vs Gânglios da Base) 
    possuem modulações AM-FM temporalmente sincronizadas.
    """

    @staticmethod
    def compute_coherence(hhs_matrix_1: np.ndarray, hhs_matrix_2: np.ndarray) -> float:
        """
        Calcula uma métrica de similaridade (Coerência Global 2D) entre duas 
        matrizes de espectro Holo-Hilbert (f_AM x f_FM) extraídas de canais diferentes
        para a mesma janela temporal (época).

        Parâmetros:
        -----------
        hhs_matrix_1 : np.ndarray
            Espectro HHS (AM x FM) do canal 1.
        hhs_matrix_2 : np.ndarray
            Espectro HHS (AM x FM) do canal 2.

        Retorna:
        --------
        coherence : float
            Valor de coerência normalizado entre [0, 1].
        """
        # Achata as matrizes para cálculo do coeficiente de correlação de Pearson
        vec_1 = hhs_matrix_1.flatten()
        vec_2 = hhs_matrix_2.flatten()

        # Evita divisão por zero se o espectro for nulo
        if np.std(vec_1) == 0 or np.std(vec_2) == 0:
            return 0.0

        # A coerência neste contexto pode ser aproximada pela correlação cruzada 
        # das topologias espectrais
        corr_matrix = np.corrcoef(vec_1, vec_2)
        coherence = corr_matrix[0, 1]

        # Retorna o valor absoluto para representar a força da coerência topológica
        return abs(float(coherence))
