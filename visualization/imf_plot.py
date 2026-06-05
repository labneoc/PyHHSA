"""
Módulo 6: visualization / imf_plot.py
Responsabilidade: Visualização em cascata das IMFs de primeira e segunda camada.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
import warnings

try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("Matplotlib não encontrado. Instale com: pip install matplotlib")
    plt = None

class IMFPlotter:
    """Ferramentas gráficas em Matplotlib para inspeção do resultado da EMD."""

    @staticmethod
    def plot_imfs(signal: np.ndarray, imfs: np.ndarray, time_axis: np.ndarray = None, title: str = "Decomposição Empírica de Modos"):
        """
        Plota o sinal original no topo e as Funções de Modo Intrínseco (IMFs) abaixo.
        """
        if plt is None:
            raise ImportError("Matplotlib necessário.")

        n_imfs = imfs.shape[0]
        if time_axis is None:
            time_axis = np.arange(len(signal))

        fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(10, 2 * (n_imfs + 1)), sharex=True)

        axes[0].plot(time_axis, signal, 'k', linewidth=1.2)
        axes[0].set_title("Sinal Original")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)

        for i in range(n_imfs):
            axes[i+1].plot(time_axis, imfs[i], 'b', linewidth=1.0)
            # O último componente costuma ser o resíduo
            label = f"IMF {i+1}" if i < n_imfs - 1 else "Resíduo"
            axes[i+1].set_ylabel(label)
            axes[i+1].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Tempo (s)")
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig
