"""
Módulo 3: hilbert / __init__.py
Responsabilidade: Transformada de Hilbert, frequências instantâneas e extração de envelopes.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

from .analytic import AnalyticSignal
from .instantaneous import InstantaneousFeatures
from .envelope import EnvelopeExtractor
from .normalization import NormalizedHilbert
