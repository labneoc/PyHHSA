"""
Módulo 6: visualization / spectrum_compare.py
Responsabilidade: Comparação visual rigorosa entre métodos espectrais.

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
    plt = None

class SpectrumComparator:
    """
    Módulo para figuras de artigos científicos.
    Permite visualizar a falha de métodos puramente aditivos (Fourier)
    em contraposição ao detalhamento não-linear da HHSA.
    """

    @staticmethod
    def plot_comparison(freqs_fft: np.ndarray, power_fft: np.ndarray, 
                        freqs_hhs: np.ndarray, power_marginal_hhs: np.ndarray):
        """
        Compara o PSD do Fourier (espalhamento espectral de sinais modulados)
        com o Espectro Marginal da HHSA.
        """
        if plt is None:
            raise ImportError("Matplotlib necessário.")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # Plot FFT
        ax1.plot(freqs_fft, power_fft, color='gray', label='Fourier (FFT)')
        ax1.set_title("Power Spectral Density - Fourier (Aditivo)")
        ax1.set_ylabel("Potência")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot Marginal HHS
        ax2.plot(freqs_hhs, power_marginal_hhs, color='blue', label='HHS Marginal (FM)')
        ax2.set_title("Espectro Marginal - HHSA (Multiplicativo e Adaptativo)")
        ax2.set_xlabel("Frequência (Hz)")
        ax2.set_ylabel("Potência")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
