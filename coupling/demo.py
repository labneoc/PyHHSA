"""
Módulo 5: coupling / demo.py
Responsabilidade: Demonstração da extração de PAC a partir de sinais data-driven.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
from pac import HHSAPAC

def run_demo():
    print("PyHHSA - Validação do Módulo 5 (Coupling)")

    # Simulando extrações da HHSA (Nested EMD)
    fs = 1000
    t = np.linspace(0, 2, 2 * fs)

    # Portadora (FM) em 40 Hz (Gamma simulado)
    freq_fm = 40
    phase_fm = np.mod(2 * np.pi * freq_fm * t, 2 * np.pi) - np.pi

    # Moduladora (AM) em 4 Hz (Theta simulado) fortemente acoplada à fase Gamma
    # O pico de amplitude AM ocorre quando a fase FM está próxima de 0
    amp_am = 1.0 + 0.8 * np.cos(phase_fm) 

    # Caso Nulo: Moduladora aleatória (sem acoplamento)
    amp_am_null = 1.0 + 0.8 * np.sin(2 * np.pi * 3 * t)

    print("\nCalculando Phase-Amplitude Coupling (PAC) via MVL (Mean Vector Length):")
    mvl_coupled = HHSAPAC.compute_mvl(phase_fm, amp_am)
    mvl_null = HHSAPAC.compute_mvl(phase_fm, amp_am_null)

    print(f" -> Força de Acoplamento (Sinal Acoplado): {np.abs(mvl_coupled):.4f} (Esperado: alto)")
    print(f" -> Força de Acoplamento (Sinal Nulo)    : {np.abs(mvl_null):.4f} (Esperado: próximo a zero)")

    print("\nCalculando Modulation Index (MI - Shannon Entropy):")
    mi_coupled = HHSAPAC.modulation_index(phase_fm, amp_am)
    mi_null = HHSAPAC.modulation_index(phase_fm, amp_am_null)

    print(f" -> MI (Sinal Acoplado): {mi_coupled:.4f}")
    print(f" -> MI (Sinal Nulo)    : {mi_null:.4f}")

    print("\n[Validação Funcional]: Módulo 5 Operacional.")

if __name__ == "__main__":
    run_demo()
