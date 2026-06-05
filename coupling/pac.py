"""
Módulo 5: coupling / pac.py
Responsabilidade: Phase-Amplitude Coupling (PAC) data-driven via HHSA.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np

class HHSAPAC:
    """
    Acoplamento Fase-Amplitude (Phase-Amplitude Coupling - PAC) utilizando
    a Transformada de Holo-Hilbert.

    Vantagem Científica:
    --------------------
    Diferente do Modulation Index (MI) de Tort ou do MVL de Canolty, 
    o PAC baseado em HHSA não requer a pré-filtragem linear do sinal em bandas
    artificiais (ex: Theta 4-8Hz, Gamma 30-80Hz). Ele utiliza as fases e amplitudes
    instantâneas extraídas diretamente das IMFs via EMD Aninhada, respeitando a
    natureza não-estacionária e não-linear da rede neural.
    """

    @staticmethod
    def compute_mvl(phase_fm: np.ndarray, amplitude_am: np.ndarray) -> complex:
        """
        Calcula o Mean Vector Length (MVL) empírico entre a fase da portadora (FM)
        e a amplitude da moduladora (AM) extraídas da HHSA.

        Parâmetros:
        -----------
        phase_fm : np.ndarray
            Série temporal da fase instantânea (radianos) da IMF portadora (Camada 1).
        amplitude_am : np.ndarray
            Série temporal da amplitude instantânea da IMF moduladora (Camada 2).

        Retorna:
        --------
        mvl_complex : complex
            Vetor complexo cujo módulo representa a força do acoplamento e o 
            ângulo representa a fase preferencial de acoplamento.
        """
        if len(phase_fm) != len(amplitude_am):
            raise ValueError("As séries temporais de fase e amplitude devem ter o mesmo tamanho.")

        # Constrói os vetores complexos: Amplitude * e^(i * Fase)
        complex_vectors = amplitude_am * np.exp(1j * phase_fm)

        # O Mean Vector Length é a média desses vetores
        mvl = np.mean(complex_vectors)
        return mvl

    @staticmethod
    def modulation_index(phase_fm: np.ndarray, amplitude_am: np.ndarray, n_bins: int = 18) -> float:
        """
        Adaptação do Modulation Index (MI) de Tort et al. (2010) para o espaço HHSA.
        Mede a divergência de Kullback-Leibler (Entropia de Shannon) da distribuição
        de amplitudes sobre os bins de fase em relação a uma distribuição uniforme.
        """
        # Define os bins de fase entre -pi e pi (ou 0 e 2pi, dependendo do input)
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

        # Calcula a amplitude média em cada bin de fase
        mean_amp = np.zeros(n_bins)
        for i in range(n_bins):
            # Encontra os índices onde a fase cai neste bin
            idx = np.where((phase_fm >= phase_bins[i]) & (phase_fm < phase_bins[i+1]))[0]
            if len(idx) > 0:
                mean_amp[i] = np.mean(amplitude_am[idx])

        # Normaliza as amplitudes para formar uma distribuição de probabilidade (P)
        sum_amp = np.sum(mean_amp)
        if sum_amp == 0:
            return 0.0

        p = mean_amp / sum_amp

        # Calcula a Entropia de Shannon (H)
        # Para lidar com log(0), mascaramos os valores onde p=0
        p_nonzero = p[p > 0]
        h = -np.sum(p_nonzero * np.log(p_nonzero))

        # Entropia máxima (distribuição uniforme)
        h_max = np.log(n_bins)

        # Modulation Index: (H_max - H) / H_max
        mi = (h_max - h) / h_max
        return mi
