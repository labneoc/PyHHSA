"""
Módulo 4: hhsa / nested_emd.py
Responsabilidade: Implementação da EMD Aninhada (Nested EMD) para duas camadas.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np

class NestedEMD:
    """
    Executa a Decomposição Empírica de Modos em duas camadas.
    Camada 1: Decompõe o sinal original nas portadoras FM (IMFs).
    Camada 2: Extrai o envelope de cada IMF e o decompõe nas moduladoras AM.
    """

    def __init__(self, fm_decomposer, am_decomposer, envelope_extractor):
        """
        Injeta as dependências dos módulos 2 (decomposition) e 3 (hilbert).

        Parâmetros:
        -----------
        fm_decomposer : objeto (ex: ICEEMDANDecomposer)
            Instância do decompositor para a 1ª camada.
        am_decomposer : objeto (ex: MaskingEMD)
            Instância do decompositor para a 2ª camada (envelopes).
        envelope_extractor : classe ou objeto (ex: EnvelopeExtractor)
            Utilitário para extração do envelope via spline.
        """
        self.fm_decomp = fm_decomposer
        self.am_decomp = am_decomposer
        self.env_ext = envelope_extractor

    def decompose(self, signal: np.ndarray) -> dict:
        """
        Processa o sinal através das duas camadas.

        Retorna:
        --------
        dict com chaves:
            'imfs_fm': np.ndarray (n_imfs_fm, tempo)
            'envelopes': np.ndarray (n_imfs_fm, tempo)
            'imfs_am': list of np.ndarray [n_imfs_fm] -> (n_imfs_am, tempo)
        """
        # Camada 1: Extração das portadoras (FM)
        imfs_fm = self.fm_decomp.decompose(signal)

        n_fm = imfs_fm.shape[0]
        n_samples = imfs_fm.shape[1]

        envelopes = np.zeros((n_fm, n_samples))
        imfs_am_list = []

        # Camada 2: Para cada portadora, extrair e decompor o envelope (AM)
        for i in range(n_fm):
            env = self.env_ext.get_absolute_envelope(imfs_fm[i, :])
            envelopes[i, :] = env

            # Decomposição do envelope para achar modulações lentas
            am_components = self.am_decomp.decompose(env)
            imfs_am_list.append(am_components)

        return {
            'imfs_fm': imfs_fm,
            'envelopes': envelopes,
            'imfs_am': imfs_am_list
        }
