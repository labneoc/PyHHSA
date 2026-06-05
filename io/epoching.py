"""
Módulo 1: io / epoching.py
Responsabilidade: Segmentação do dado contínuo baseada em eventos ou épocas de tamanho fixo (resting state).

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import mne
import numpy as np
from typing import Optional, Dict, Tuple

class NeuralEpoching:
    """
    Extrai matrizes tridimensionais (Épocas x Canais x Tempo) do sinal bruto.
    """

    @staticmethod
    def extract_event_epochs(raw: mne.io.Raw, 
                             events: np.ndarray, 
                             event_id: Dict[str, int], 
                             tmin: float, 
                             tmax: float, 
                             baseline: Optional[Tuple[float, float]] = (None, 0)) -> mne.Epochs:
        """
        Segmenta o sinal com base em uma matriz de eventos (triggers).
        
        Parâmetros:
        -----------
        raw : mne.io.Raw
            Objeto do sinal contínuo
        events : np.ndarray
            Matriz de eventos extraída (mne.find_events)
        event_id : dict
            Dicionário mapeando os nomes das condições para os triggers (ex: {'Mão_Direita': 1})
        tmin / tmax : float
            Início e fim da época em relação ao trigger (em segundos)
        baseline : tuple
            Intervalo de correção de linha de base. (None, 0) significa do início da época até o trigger.
        """
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, 
                            baseline=baseline, preload=True, verbose='ERROR')
        return epochs

    @staticmethod
    def extract_resting_state(raw: mne.io.Raw, duration: float = 2.0, overlap: float = 0.0) -> mne.Epochs:
        """
        Cria épocas artificiais de tamanho fixo, ideal para análises de 
        estado de repouso (resting state) ou sinais contínuos sem task específica.
        """
        events = mne.make_fixed_length_events(raw, id=1, duration=duration, overlap=overlap)
        epochs = mne.Epochs(raw, events, event_id={'Rest': 1}, tmin=0., tmax=duration,
                            baseline=None, preload=True, verbose='ERROR')
        return epochs