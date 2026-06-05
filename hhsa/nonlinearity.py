"""
Módulo 4: hhsa / nonlinearity.py
Responsabilidade: Cálculo do Índice de Não-Linearidade Inter-Modo.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np

class NonlinearityIndex:
    """
    Quantifica o grau de acoplamento multiplicativo entre diferentes escalas 
    (cross-scale phase-locked modulation), uma medida central de sistemas complexos.
    O índice N_in varia de 0 (estritamente linear/aditivo) até 1 (altamente não linear).
    Baseado em Huang et al. (2016).
    """

    @staticmethod
    def calculate_inter_mode(envelopes: np.ndarray) -> float:
        """
        Calcula o índice N_in a partir dos envelopes extraídos na Camada 1.

        Fórmula: N_in = std(a_j(t)) / rms(a_j(t))
        A razão entre a soma das variâncias dos envelopes sobre a energia total.

        Parâmetros:
        -----------
        envelopes : np.ndarray (n_imfs, tempo)
            Matriz de envelopes extraídos das IMFs (camada 1).

        Retorna:
        --------
        n_in : float
            Índice entre 0 e 1.
        """
        # Exclui o resíduo final (última linha) do cálculo se desejado, 
        # mas aqui usaremos toda a matriz de modo geral.

        # Variância (flutuação) de cada envelope no tempo
        variances = np.var(envelopes, axis=1)

        # Energia total (Root Mean Square ao quadrado) de cada envelope
        mean_squares = np.mean(envelopes**2, axis=1)

        sum_variances = np.sum(variances)
        sum_mean_squares = np.sum(mean_squares)

        if sum_mean_squares == 0:
            return 0.0

        n_in = sum_variances / sum_mean_squares

        # Garante limite superior teórico em implementações discretas
        return min(float(n_in), 1.0)
