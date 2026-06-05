"""
Módulo 7: features / mean_freq.py
Responsabilidade: Cálculo do centroide de frequência (Mean Frequency) da portadora e da moduladora.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np

class MeanFrequency:
    """
    A Frequência Média Ponderada pela Amplitude (Mean Frequency) descreve a 
    frequência dominante exata (cf_FM e cf_AM) numa dada janela temporal.
    Na HHSA aplicada ao motor-cortex, o centroide de alpha shift para frequências 
    mais baixas durante o movimento lento vs. movimento rápido.
    """

    @staticmethod
    def calculate_cf_fm(spectrum: np.ndarray, fm_edges: np.ndarray) -> float:
        """
        Centroide da portadora (cf_FM).
        É a Frequência FM média ponderada pelo espectro marginal.
        """
        marginal_fm = np.sum(spectrum, axis=0) # (1D ao longo do eixo FM)
        fm_centers = (fm_edges[:-1] + fm_edges[1:]) / 2

        total_power = np.sum(marginal_fm)
        if total_power == 0:
            return 0.0

        cf_fm = np.sum(fm_centers * marginal_fm) / total_power
        return float(cf_fm)

    @staticmethod
    def calculate_cf_am(spectrum: np.ndarray, am_edges: np.ndarray) -> float:
        """
        Centroide da moduladora (cf_AM).
        É a Frequência AM média ponderada pelo espectro marginal.
        """
        marginal_am = np.sum(spectrum, axis=1) # (1D ao longo do eixo AM)
        am_centers = (am_edges[:-1] + am_edges[1:]) / 2

        total_power = np.sum(marginal_am)
        if total_power == 0:
            return 0.0

        cf_am = np.sum(am_centers * marginal_am) / total_power
        return float(cf_am)
