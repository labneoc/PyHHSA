"""
Módulo 3: hilbert / demo.py
Responsabilidade: Validação técnica da extração de parâmetros instantâneos e NHT.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
from analytic import AnalyticSignal
from instantaneous import InstantaneousFeatures
from envelope import EnvelopeExtractor
from normalization import NormalizedHilbert

def run_demo():
    print("PyHHSA - Validação do Módulo 3 (Hilbert & Envelopes)")
    fs = 1000
    t = np.linspace(0, 2, fs*2)

    # Cria uma IMF sintética: 10 Hz portadora com envelope logístico (não-linear)
    am_env = 2.0 / (1.0 + np.exp(-5*(t - 1.0)))  # Sigmoide
    carrier = np.sin(2 * np.pi * 10 * t)
    imf = am_env * carrier

    print("\n1. Cálculo Clássico de Hilbert...")
    z = AnalyticSignal.compute(imf)
    ia = InstantaneousFeatures.extract_ia(z)
    freq = InstantaneousFeatures.extract_if(z, fs)
    print(f" -> Amplitude Média (Hilbert Clássico): {np.mean(ia):.3f}")
    print(f" -> Frequência Média (Hilbert Clássico): {np.mean(freq):.3f} Hz")

    print("\n2. Extração de Envelope via Splines (Base para HHSA Layer 2)...")
    spline_env = EnvelopeExtractor.get_absolute_envelope(imf)
    print(f" -> Correlação Envelope Spline vs Envelope Teórico: {np.corrcoef(spline_env, am_env)[0,1]:.4f}")

    print("\n3. Transformada de Hilbert Normalizada (NHT)...")
    F, A = NormalizedHilbert.normalize(imf, max_iter=3)
    print(f" -> Amplitude Máxima da portadora FM normalizada: {np.max(np.abs(F)):.4f} (Desejado: 1.0)")
    print(f" -> Correlação Envelope NHT vs Envelope Teórico: {np.corrcoef(A, am_env)[0,1]:.4f}")

    print("\n[Validação Funcional]: Módulo 3 Operacional.")

if __name__ == "__main__":
    run_demo()
