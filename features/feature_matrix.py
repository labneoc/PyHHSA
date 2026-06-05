"""
Módulo 7: features / feature_matrix.py
Responsabilidade: Consolidação e empacotamento das métricas em um DataFrame.

Prof. Dr. Bruno Duarte Gomes
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA
"""

import warnings
import numpy as np

try:
    import pandas as pd
except ImportError:
    warnings.warn("Pandas não encontrado. FeatureMatrixBuilder dependerá apenas de dicionários.")
    pd = None

class FeatureMatrixBuilder:
    """
    Fábrica que coleta os resultados do pipeline (P_total, cf_FM, cf_AM, PAC_mvl, N_in)
    para múltiplos trials e canais, gerando a matriz X e o vetor Y padronizados 
    para o ecossistema Scikit-Learn ou PyTorch.
    """

    def __init__(self):
        self.trials = []

    def add_trial(self, subject_id: str, label: int, features: dict):
        """
        Adiciona uma época decodificada ao banco.

        Parâmetros:
        -----------
        subject_id : str
            Identificador (ex: "S01"). Útil para validação Leave-One-Subject-Out.
        label : int
            A classe (Y) a ser predita (ex: 0 = Repouso, 1 = Intenção Motora D).
        features : dict
            Dicionário achatado das features.
            Ex: {'C3_Alpha_Power': 14.5, 'Cz_N_in': 0.45}
        """
        row = features.copy()
        row['subject_id'] = subject_id
        row['label'] = label
        self.trials.append(row)

    def build_dataframe(self):
        """Retorna um DataFrame Pandas com todo o dataset empilhado (se disponível)."""
        if pd is None:
            raise ImportError("Pandas necessário para esta função.")
        return pd.DataFrame(self.trials)

    def export_sklearn_format(self):
        """
        Gera as matrizes X (features) e y (rótulos) prontas para o ML.

        Retorna:
        --------
        X : np.ndarray (n_trials, n_features)
        y : np.ndarray (n_trials,)
        feature_names : list of str
        """
        if pd is None:
            raise ImportError("Pandas necessário para esta função.")

        df = self.build_dataframe()
        y = df['label'].values

        # Remove as colunas de metadados
        drop_cols = ['subject_id', 'label']
        X_df = df.drop(columns=drop_cols, errors='ignore')

        X = X_df.values
        feature_names = X_df.columns.tolist()

        return X, y, feature_names
