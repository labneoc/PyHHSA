"""
Módulo 7: features / __init__.py
Responsabilidade: Extração quantitativa de features para pipelines de Brain Decoding / Machine Learning.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

from .roi_power import ROIPower
from .mean_freq import MeanFrequency
from .nonlin_index import NonlinearityFeature
from .feature_matrix import FeatureMatrixBuilder
