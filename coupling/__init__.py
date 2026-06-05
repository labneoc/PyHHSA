"""
Módulo 5: coupling / __init__.py
Responsabilidade: Análise de acoplamento de fase-amplitude e conectividade cross-scale.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

from .pac import HHSAPAC
from .coherence import HHSACoherence
from .connectivity import FunctionalConnectivity
