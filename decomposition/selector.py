"""
Módulo 2: decomposition / selector.py
Responsabilidade: Fábrica (Factory) para seleção dinâmica do melhor método de decomposição.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

from .emd import EMDDecomposer
from .ceemdan import CEEMDANDecomposer
from .iceemdan import ICEEMDANDecomposer
from .vmd import VMDDecomposer
from .masking_emd import MaskingEMD

class DecompositionSelector:
    """
    Fábrica que retorna a instância correta do algoritmo, garantindo o encapsulamento.
    Na HHSA, permite chaveamento automático dependendo da camada (ex: CEEMDAN na 1ª, Masking na 2ª).
    """

    @staticmethod
    def get_decomposer(method: str, **kwargs):
        """
        Retorna o objeto decompositor solicitado.

        Parâmetros:
        -----------
        method : str
            'EMD', 'CEEMDAN', 'ICEEMDAN', 'VMD', 'MASKING'
        **kwargs :
            Hiperparâmetros passados ao construtor da classe respectiva.
        """
        method = method.upper()
        if method == 'EMD':
            return EMDDecomposer(**kwargs)
        elif method == 'CEEMDAN':
            return CEEMDANDecomposer(**kwargs)
        elif method == 'ICEEMDAN':
            return ICEEMDANDecomposer(**kwargs)
        elif method == 'VMD':
            return VMDDecomposer(**kwargs)
        elif method == 'MASKING':
            return MaskingEMD(**kwargs)
        else:
            raise ValueError(f"Método de decomposição '{method}' não suportado na PyHHSA.")
