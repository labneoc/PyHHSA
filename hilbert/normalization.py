"""
Módulo 3: hilbert / normalization.py
Responsabilidade: Normalized Hilbert Transform (NHT) para alta não-linearidade.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
from .envelope import EnvelopeExtractor

class NormalizedHilbert:
    """
    A Normalized Hilbert Transform (Huang et al., 2009) isola o conteúdo FM (Frequency Modulated)
    e o conteúdo AM (Amplitude Modulated) empiricamente antes de aplicar a Transformada de Hilbert.
    Essencial para obter Frequências Instantâneas com significado físico em IMFs altamente não-lineares,
    evitando violações do Teorema de Bedrosian.
    """

    @staticmethod
    def normalize(imf: np.ndarray, max_iter: int = 5, tol: float = 0.01) -> tuple:
        """
        Divide a IMF pelo seu envelope interativamente até que a amplitude máxima
        da portadora FM (F(t)) seja estritamente igual a 1 (ou dentro da tolerância).

        Retorna:
        --------
        F : np.ndarray
            O componente puramente modulado em frequência (|F(t)| <= 1).
        A : np.ndarray
            O envelope de amplitude (AM) extraído.
        """
        F = np.copy(imf)
        A = np.ones_like(imf)

        for _ in range(max_iter):
            # Extrai envelope da portadora atual
            env = EnvelopeExtractor.get_absolute_envelope(F)

            # Atualiza o envelope AM total
            A = A * env

            # Normaliza a portadora
            F = F / env

            # Critério de parada: Se todos os valores estão contidos em [-1-tol, 1+tol]
            if np.all(np.abs(F) <= 1.0 + tol):
                break

        # Força restrição estrita [-1, 1] devido a problemas numéricos nas bordas
        F = np.clip(F, -1.0, 1.0)

        return F, A
