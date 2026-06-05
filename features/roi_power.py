"""
Módulo 7: features / roi_power.py
Responsabilidade: Extração de Potência Total (P_total) por Região de Interesse (ROI) Espectral.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np

class ROIPower:
    """
    Integra a energia da matriz de Holo-Hilbert dentro de caixas de frequência
    específicas (Region of Interest espectral). Na literatura de HHSA (Hsu 2018), 
    usualmente extraímos a energia de uma portadora FM (ex: beta) sendo modulada 
    por uma frequência lenta AM.
    """

    @staticmethod
    def extract_power_in_band(spectrum: np.ndarray, fm_edges: np.ndarray, am_edges: np.ndarray, 
                              fm_band: tuple, am_band: tuple = (0.0, float('inf'))) -> float:
        """
        Calcula a potência agregada em uma ROI espectral (f_FM, f_AM).

        Parâmetros:
        -----------
        spectrum : np.ndarray
            Matriz HHSA 2D. Eixo Y é AM, Eixo X é FM.
        fm_edges : np.ndarray
            Fronteiras dos bins de frequência portadora (FM).
        am_edges : np.ndarray
            Fronteiras dos bins de frequência moduladora (AM).
        fm_band : tuple
            Tupla (min_Hz, max_Hz) da portadora (ex: Alpha = (8, 13)).
        am_band : tuple
            Tupla (min_Hz, max_Hz) da moduladora (ex: Delta AM = (0.1, 4)).

        Retorna:
        --------
        total_power : float
            Soma da energia na ROI selecionada.
        """
        # Encontra os índices válidos da matriz
        fm_valid_idx = np.where((fm_edges[:-1] >= fm_band[0]) & (fm_edges[1:] <= fm_band[1]))[0]
        am_valid_idx = np.where((am_edges[:-1] >= am_band[0]) & (am_edges[1:] <= am_band[1]))[0]

        if len(fm_valid_idx) == 0 or len(am_valid_idx) == 0:
            return 0.0

        # Pega a submatriz da ROI
        roi_matrix = spectrum[np.ix_(am_valid_idx, fm_valid_idx)]

        return float(np.sum(roi_matrix))
