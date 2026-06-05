"""
Módulo 6: visualization / demo.py
Responsabilidade: Validação da infraestrutura gráfica do módulo.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np

def run_demo():
    print("PyHHSA - Validação do Módulo 6 (Visualization)")

    print("Tentando importar as classes gráficas...")
    try:
        from hhs_plot import HoloHilbertPlotter
        from imf_plot import IMFPlotter
        from spectrum_compare import SpectrumComparator
        from topomap import HHSATopomap

        print("Classes importadas com sucesso.")

        # Simula uma matriz HHSA reduzida
        print("Testando instanciar figura no Plotly (HHS Matrix)...")
        fm_edges = np.linspace(0, 50, 11)
        am_edges = np.linspace(0, 10, 6)
        matrix = np.random.rand(5, 10) # 5 bins AM, 10 bins FM

        fig_plotly = HoloHilbertPlotter.plot_heatmap_2d(matrix, fm_edges, am_edges)

        print("Testando instanciar figura no Matplotlib (IMF Plot)...")
        time_ax = np.linspace(0, 1, 100)
        sig = np.sin(2 * np.pi * 5 * time_ax)
        imfs = np.array([sig]) # Uma IMF fake

        fig_mpl = IMFPlotter.plot_imfs(sig, imfs, time_ax)

        print("\n[Validação Funcional]: Módulo 6 Operacional.")
        print("(As figuras não foram exibidas para não bloquear a execução do terminal)")

    except Exception as e:
        print(f"Erro ao instanciar os gráficos: {e}")
        print("Verifique dependências: matplotlib, plotly e mne.")

if __name__ == "__main__":
    run_demo()
