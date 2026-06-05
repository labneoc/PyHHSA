"""
Módulo 3: hilbert / instantaneous.py
Responsabilidade: Estimação de Frequência Instantânea (IF) e Amplitude Instantânea (IA).

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np

class InstantaneousFeatures:
    """
    Extrai a Amplitude Instantânea (IA) e a Frequência Instantânea (IF) 
    a partir do sinal analítico complexo z(t).
    """

    @staticmethod
    def extract_ia(analytic_signal: np.ndarray) -> np.ndarray:
        """
        Amplitude Instantânea: a(t) = |z(t)|.
        """
        return np.abs(analytic_signal)

    @staticmethod
    def extract_phase(analytic_signal: np.ndarray) -> np.ndarray:
        """
        Fase Instantânea: theta(t) = arctan(Im(z) / Re(z)).
        Utiliza np.unwrap para garantir a continuidade da fase (sem saltos de 2pi).
        """
        phase = np.angle(analytic_signal)
        return np.unwrap(phase, axis=-1)

    @staticmethod
    def extract_if(analytic_signal: np.ndarray, fs: float) -> np.ndarray:
        """
        Frequência Instantânea: f(t) = (1 / 2*pi) * d(theta)/dt.

        Parâmetros:
        -----------
        analytic_signal : np.ndarray
            Sinal complexo.
        fs : float
            Frequência de amostragem em Hz.

        Retorna:
        --------
        inst_freq : np.ndarray
            Frequência em Hz para cada ponto de tempo.
        """
        phase = InstantaneousFeatures.extract_phase(analytic_signal)

        # d(theta)/dt usando gradiente central para maior estabilidade numérica
        d_phase = np.gradient(phase, axis=-1)

        inst_freq = (d_phase * fs) / (2.0 * np.pi)

        # Correção para frequências negativas residuais devido a artefatos numéricos
        inst_freq[inst_freq < 0] = 0.0

        return inst_freq
