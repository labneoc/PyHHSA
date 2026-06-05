"""
Módulo 3: hilbert / envelope.py
Responsabilidade: Extração de envelope por spline natural.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

class EnvelopeExtractor:
    """
    A HHSA (Huang et al., 2016) exige a extração de envelopes das IMFs portadoras (camada 1)
    para que estes possam ser submetidos à segunda camada de decomposição (EMD aninhada).
    Este processo é fundamental para evidenciar a modulação multiplicativa inter-escala.
    """

    @staticmethod
    def get_upper_envelope(signal: np.ndarray) -> np.ndarray:
        """
        Calcula o envelope superior passando um spline cúbico pelos máximos locais.
        """
        # Identifica os índices dos máximos locais
        peaks, _ = find_peaks(signal)

        if len(peaks) < 2:
            return np.ones_like(signal) * np.max(signal)

        # Adiciona extremidades para ancorar o spline e evitar Efeito Runge (oscilações nas bordas)
        t = np.arange(len(signal))
        peaks_ext = np.concatenate(([0], peaks, [len(signal)-1]))
        vals_ext = np.concatenate(([signal[0]], signal[peaks], [signal[-1]]))

        cs = CubicSpline(peaks_ext, vals_ext, bc_type='natural')
        return cs(t)

    @staticmethod
    def get_absolute_envelope(signal: np.ndarray) -> np.ndarray:
        """
        Na HHSA aplicada a dados não lineares, o envelope AM costuma ser 
        calculado sobre o valor absoluto da IMF |x(t)|.
        """
        return EnvelopeExtractor.get_upper_envelope(np.abs(signal))
