"""
Módulo 4: hhsa / hhs_builder.py
Responsabilidade: Construção do Espectro Holo-Hilbert.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np

class HoloHilbertSpectrum:
    """
    Mapeia os vetores de frequência instantânea (FM e AM) e amplitude instantânea 
    em uma grade bidimensional (f_FM, f_AM), acumulando energia. 
    Integra sobre o tempo para gerar o espectro estático AM-FM.
    """

    def __init__(self, fm_bins: int = 100, am_bins: int = 50, 
                 fm_range: tuple = (0, 100), am_range: tuple = (0, 20)):
        self.fm_bins = fm_bins
        self.am_bins = am_bins
        self.fm_range = fm_range
        self.am_range = am_range

    def build_spectrum(self, if_fm: list, if_am: list, ia_am: list) -> np.ndarray:
        """
        Constrói o espectro 2D (f_FM no eixo X, f_AM no eixo Y).

        Parâmetros:
        -----------
        if_fm : list of np.ndarray
            Frequências instantâneas das portadoras (Camada 1).
        if_am : list of list of np.ndarray
            Frequências instantâneas das moduladoras (Camada 2).
        ia_am : list of list of np.ndarray
            Amplitudes instantâneas (ou energia) das moduladoras (Camada 2).

        Retorna:
        --------
        spectrum : np.ndarray (am_bins, fm_bins)
            Matriz espectral com densidade de energia acumulada.
        """
        spectrum = np.zeros((self.am_bins, self.fm_bins))

        fm_edges = np.linspace(self.fm_range[0], self.fm_range[1], self.fm_bins + 1)
        am_edges = np.linspace(self.am_range[0], self.am_range[1], self.am_bins + 1)

        # Loop sobre cada portadora (Camada 1)
        for i, fm_freqs in enumerate(if_fm):
            # Loop sobre as moduladoras desta portadora (Camada 2)
            for j, am_freqs in enumerate(if_am[i]):
                am_amps = ia_am[i][j]

                # Encontra os índices (bins) correspondentes para FM e AM
                # np.digitize retorna índices começando em 1, subtraímos 1 para usar como índice do array
                fm_idx = np.digitize(fm_freqs, fm_edges) - 1
                am_idx = np.digitize(am_freqs, am_edges) - 1

                # Acumula energia pixel a pixel no espaço de fases (f_FM, f_AM)
                for t_idx in range(len(fm_freqs)):
                    f_i = fm_idx[t_idx]
                    a_i = am_idx[t_idx]

                    # Verifica limites
                    if 0 <= f_i < self.fm_bins and 0 <= a_i < self.am_bins:
                        # Energia proporcional ao quadrado da amplitude
                        spectrum[a_i, f_i] += am_amps[t_idx] ** 2

        return spectrum
