"""
Módulo 7: features / nonlin_index.py
Responsabilidade: Wrapper para extrair índices de não-linearidade como uma feature escalonada.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np

class NonlinearityFeature:
    """
    Obtém o grau de não-linearidade (intra e inter) para enriquecer o vetor
    de características que será enviado ao SVM/Random Forest. Redes biológicas
    complexas mudam seu N_in dependendo do estado cognitivo.
    """

    @staticmethod
    def extract_inter_mode(envelopes: np.ndarray) -> float:
        """
        Calcula a razão da variância AM (flutuações da modulação) vs Energia AM Total.
        (N_in). Requer acesso às matrizes de envelopes calculadas na Fase 1 do Nested EMD.
        """
        variances = np.var(envelopes, axis=1)
        mean_squares = np.mean(envelopes**2, axis=1)

        sum_variances = np.sum(variances)
        sum_mean_squares = np.sum(mean_squares)

        if sum_mean_squares == 0:
            return 0.0

        return min(float(sum_variances / sum_mean_squares), 1.0)
