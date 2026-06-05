"""
Módulo 2: decomposition / __init__.py
Responsabilidade: Exportação dos algoritmos de decomposição da suite PyHHSA.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

from .emd import EMDDecomposer
from .ceemdan import CEEMDANDecomposer
from .iceemdan import ICEEMDANDecomposer
from .masking_emd import MaskingEMD
from .vmd import VMDDecomposer
from .selector import DecompositionSelector
