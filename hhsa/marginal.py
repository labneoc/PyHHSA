"""
Módulo 4: hhsa / marginal.py
Responsabilidade: Funções auxiliares para espectros marginais.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np

class MarginalSpectrum:
    """Implementa o colapso dimensional do Espectro Holo-Hilbert."""
    @staticmethod
    def integrate_time(hhs_3d: np.ndarray) -> np.ndarray:
        """Reduz um espectro (Tempo, AM, FM) para (AM, FM)."""
        return np.sum(hhs_3d, axis=0)
