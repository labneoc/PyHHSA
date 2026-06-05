"""
Módulo 1: io / preprocess.py
Responsabilidade: Limpeza de sinal, rereferenciamento, filtragem temporal e espacial.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import mne
import numpy as np
from typing import Optional, Union, List

class NeuralPreprocessor:
    """
    Classe contendo métodos estáticos para limpeza e preparação do sinal EEG/LFP
    antes da aplicação da Decomposição Empírica de Modos (EMD).
    """

    @staticmethod
    def apply_bandpass(raw: mne.io.Raw, l_freq: float, h_freq: float, n_jobs: int = 1) -> mne.io.Raw:
        """
        Aplica filtro FIR passa-banda de fase zero para não distorcer as relações de fase.
        Essencial antes da análise baseada em Hilbert.
        """
        # copy() é usado para manter imutabilidade do dado original na pipeline se necessário
        raw_filtered = raw.copy().filter(
            l_freq=l_freq, 
            h_freq=h_freq, 
            method='fir',
            phase='zero',
            fir_window='hamming',
            fir_design='firwin',
            n_jobs=n_jobs,
            verbose='ERROR'
        )
        return raw_filtered

    @staticmethod
    def apply_notch(raw: mne.io.Raw, freqs: Union[float, List[float]] = 60.0) -> mne.io.Raw:
        """Remove ruído de rede elétrica (60Hz no Brasil / 50Hz Europa)."""
        return raw.copy().notch_filter(freqs=freqs, method='fir', phase='zero', verbose='ERROR')

    @staticmethod
    def set_reference(raw: mne.io.Raw, ref_channels: Union[str, List[str]] = 'average') -> mne.io.Raw:
        """
        Rereferencia os dados. 
        'average' (Common Average Reference - CAR) é recomendado para alta densidade,
        mas pode causar espalhamento de fase em montagens com poucos canais.
        """
        raw_ref, _ = mne.set_eeg_reference(raw.copy(), ref_channels=ref_channels, verbose='ERROR')
        return raw_ref
        
    @staticmethod
    def apply_ica(raw: mne.io.Raw, n_components: int = 15, random_state: int = 42) -> mne.preprocessing.ICA:
        """
        Calcula a Análise de Componentes Independentes (ICA) usando o algoritmo FastICA.
        Retorna o objeto ICA ajustado. A remoção dos componentes (ex: piscadas) 
        deve ser feita selecionando os componentes espacialmente.
        """
        ica = mne.preprocessing.ICA(n_components=n_components, method='fastica', random_state=random_state)
        ica.fit(raw, verbose='ERROR')
        return ica