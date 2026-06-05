"""
Módulo 2: decomposition / demo.py
Responsabilidade: Validação técnica e demonstração de uso dos algoritmos.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
import matplotlib.pyplot as plt
from emd import EMDDecomposer
from iceemdan import ICEEMDANDecomposer

def generate_am_fm_signal(fs=1000, t_end=2.0):
    """
    Gera um sinal complexo com não-linearidade multiplicativa (AM e FM simultâneas),
    que é o padrão-ouro (ground truth) para testar métodos propostos para a HHSA.
    """
    t = np.linspace(0, t_end, int(fs * t_end), endpoint=False)

    # Portadora FM (frequência varia ao longo do tempo de 10 a 50Hz)
    inst_freq = np.linspace(10, 50, len(t))
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    carrier = np.sin(phase)

    # Moduladora AM (envelope de 2 Hz)
    modulator = 1.0 + 0.5 * np.sin(2 * np.pi * 2 * t)

    # Sinal final: Produto de Moduladora x Portadora + Ruído Branco
    noise = np.random.normal(0, 0.1, len(t))
    signal = (modulator * carrier) + noise

    return t, signal, modulator, carrier

def run_demo():
    print("PyHHSA - Validação do Módulo 2 (Decomposition)")
    print("Gerando sinal multiplicativo AM-FM sintético...")
    t, signal, mod, carrier = generate_am_fm_signal()

    try:
        print("\nExecutando EMD Clássica...")
        emd = EMDDecomposer(max_imfs=5)
        imfs_emd = emd.decompose(signal)
        print(f"-> EMD extraiu {imfs_emd.shape[0]} componentes.")

        print("\nExecutando ICEEMDAN (Colominas, 2014)...")
        # Trials baixos na demo para ser rápido
        iceemdan = ICEEMDANDecomposer(trials=10, noise_scale=0.2, max_imfs=5) 
        imfs_ice = iceemdan.decompose(signal)
        print(f"-> ICEEMDAN extraiu {imfs_ice.shape[0]} componentes.")

        print("\n[Validação Funcional]: Módulo 2 Operacional.")

    except Exception as e:
        print(f"Erro durante a execução dos algoritmos numéricos: {e}")
        print("Certifique-se de que a biblioteca 'EMD-signal' está instalada no seu ambiente virtual.")

if __name__ == "__main__":
    run_demo()
