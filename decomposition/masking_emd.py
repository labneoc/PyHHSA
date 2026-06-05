"""
Módulo 2: decomposition / masking_emd.py
Responsabilidade: Masking EMD (Essencial para a 2ª Camada da HHSA).

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
from .emd import EMDDecomposer

class MaskingEMD:
    """
    Masking EMD adiciona uma onda senoidal (máscara) de alta frequência ao sinal
    antes da decomposição, forçando a extração de componentes limpos e evitando mode-mixing.
    Fundamental para a decomposição dos envelopes (AM) na 2ª camada da HHSA.
    """

    def __init__(self, mask_freq: float, mask_amp: float, fs: float):
        self.mask_freq = mask_freq
        self.mask_amp = mask_amp
        self.fs = fs
        self.emd_solver = EMDDecomposer(max_imfs=5) # 2ª camada costuma ter menos IMFs

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        time_axis = np.arange(len(signal)) / self.fs

        # Cria a máscara senoidal
        mask = self.mask_amp * np.sin(2 * np.pi * self.mask_freq * time_axis)

        # Decomposição em fase (Sinal + Máscara)
        imfs_pos = self.emd_solver.decompose(signal + mask, time_axis)

        # Decomposição em contrafase (Sinal - Máscara)
        imfs_neg = self.emd_solver.decompose(signal - mask, time_axis)

        # Alinha a quantidade de IMFs (pega o mínimo entre os dois)
        min_imfs = min(imfs_pos.shape[0], imfs_neg.shape[0])

        # Calcula a média das IMFs. Como as máscaras estão em oposição de fase, 
        # a média linearmente anula a máscara e preserva o conteúdo do sinal sem mode-mixing.
        imfs_mean = (imfs_pos[:min_imfs, :] + imfs_neg[:min_imfs, :]) / 2.0

        return imfs_mean
