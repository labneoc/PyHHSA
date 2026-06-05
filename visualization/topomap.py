"""
Módulo 6: visualization / topomap.py
Responsabilidade: Topografia cerebral de métricas derivadas da HHSA integrando MNE.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
import warnings

try:
    import mne
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("MNE ou Matplotlib não encontrados.")
    mne = None
    plt = None

class HHSATopomap:
    """
    Projeta as extrações quantitativas da HHSA (como energia de acoplamento AM/FM)
    no espaço topográfico do escalpo usando a infraestrutura do MNE-Python.
    """

    @staticmethod
    def plot_power_topomap(values: np.ndarray, info: 'mne.Info', title: str = "HHS Power Topomap"):
        """
        Desenha o topomap do EEG.

        Parâmetros:
        -----------
        values : np.ndarray
            Vetor 1D contendo o valor da métrica (ex: PAC, ou Potência AM) para cada canal.
        info : mne.Info
            Objeto MNE contendo as posições (montagem) dos eletrodos.
        """
        if mne is None or plt is None:
            raise ImportError("MNE e Matplotlib são necessários para topomaps.")

        fig, ax = plt.subplots(figsize=(5, 5))

        # Dependendo da versão do MNE, a API muda ligeiramente. 
        # A forma recomendada atual é mne.viz.plot_topomap
        im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap='Spectral_r')

        cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
        cbar.set_label('Potência / Índice')
        ax.set_title(title)

        return fig
