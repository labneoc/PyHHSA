"""
Módulo 6: visualization / hhs_plot.py
Responsabilidade: Visualização 3D e 2D interativa do Espectro HHSA via Plotly.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import numpy as np
import warnings

try:
    import plotly.graph_objects as go
except ImportError:
    warnings.warn("Plotly não encontrado. Instale com: pip install plotly")
    go = None

class HoloHilbertPlotter:
    """
    Renderiza o espectro Holo-Hilbert (AM x FM x Energia) em ambientes interativos.
    Essencial para explorar visualmente o acoplamento cruzado de escalas.
    """

    @staticmethod
    def plot_surface_3d(spectrum_matrix: np.ndarray, fm_edges: np.ndarray, am_edges: np.ndarray, title: str = "Holo-Hilbert Spectrum 3D") -> 'go.Figure':
        """
        Cria um gráfico de superfície 3D (Surface Plot).
        Eixo X: Frequência FM (Portadora, ex: bandas rápidas Gamma/High-Gamma)
        Eixo Y: Frequência AM (Moduladora, ex: bandas lentas Theta/Alpha)
        Eixo Z: Potência Espectral
        """
        if go is None:
            raise ImportError("Biblioteca Plotly é necessária para esta função.")

        # Calcular os centros dos bins para plotagem
        fm_centers = (fm_edges[:-1] + fm_edges[1:]) / 2
        am_centers = (am_edges[:-1] + am_edges[1:]) / 2

        fig = go.Figure(data=[go.Surface(
            z=spectrum_matrix,
            x=fm_centers,
            y=am_centers,
            colorscale='Viridis',
            colorbar=dict(title='Potência')
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Frequência Portadora (FM) [Hz]',
                yaxis_title='Frequência Moduladora (AM) [Hz]',
                zaxis_title='Potência'
            ),
            width=800,
            height=600,
            template="plotly_dark"
        )
        return fig

    @staticmethod
    def plot_heatmap_2d(spectrum_matrix: np.ndarray, fm_edges: np.ndarray, am_edges: np.ndarray, title: str = "Holo-Hilbert Heatmap") -> 'go.Figure':
        """
        Gera um mapa de calor (vista superior do espectro), ideal para publicações
        impressas quando exportado em alta resolução.
        """
        if go is None:
            raise ImportError("Biblioteca Plotly é necessária para esta função.")

        fm_centers = (fm_edges[:-1] + fm_edges[1:]) / 2
        am_centers = (am_edges[:-1] + am_edges[1:]) / 2

        fig = go.Figure(data=go.Heatmap(
            z=spectrum_matrix,
            x=fm_centers,
            y=am_centers,
            colorscale='Jet',
            colorbar=dict(title='Potência')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Frequência Portadora (FM) [Hz]',
            yaxis_title='Frequência Moduladora (AM) [Hz]',
            width=800,
            height=600,
            template="plotly_dark"
        )
        return fig
