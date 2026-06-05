"""
Módulo 1: io / reader.py
Responsabilidade: Leitura e padronização de formatos eletrofisiológicos (EDF, BrainVision, FIFF, EEGLAB).

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import mne
import os
from typing import Optional, List

class NeuralReader:
    """
    Classe para importação e padronização de sinais neurais.
    Converte múltiplos formatos físicos em um objeto mne.io.Raw na memória.
    """
    
    @staticmethod
    def load_edf(file_path: str, preload: bool = True, exclude: Optional[List[str]] = None) -> mne.io.Raw:
        """
        Carrega dados em formato EDF/BDF.
        
        Parâmetros:
        -----------
        file_path : str
            Caminho para o arquivo .edf ou .bdf
        preload : bool
            Se True, carrega os dados para a memória RAM (necessário para filtragem).
        exclude : List[str], opcional
            Canais a serem ignorados durante a leitura para economizar memória.
            
        Retorna:
        --------
        raw : mne.io.Raw
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
        exclude_channels = exclude if exclude is not None else ()
        raw = mne.io.read_raw_edf(file_path, preload=preload, exclude=exclude_channels, verbose='ERROR')
        return raw

    @staticmethod
    def load_brainvision(vhdr_path: str, preload: bool = True) -> mne.io.Raw:
        """Carrega dados em formato BrainVision (.vhdr)."""
        if not os.path.exists(vhdr_path):
            raise FileNotFoundError(f"Arquivo VHDR não encontrado: {vhdr_path}")
        return mne.io.read_raw_brainvision(vhdr_path, preload=preload, verbose='ERROR')
        
    @staticmethod
    def load_eeglab(set_path: str, preload: bool = True) -> mne.io.Raw:
        """Carrega dados pré-processados no formato EEGLAB (.set)."""
        if not os.path.exists(set_path):
            raise FileNotFoundError(f"Arquivo SET não encontrado: {set_path}")
        return mne.io.read_raw_eeglab(set_path, preload=preload, verbose='ERROR')