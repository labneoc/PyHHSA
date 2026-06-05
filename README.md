# PyHHSA — Holo-Hilbert Spectral Analysis Suite for Electrophysiology

![Status](https://img.shields.io/badge/status-active-success)  ![License](https://img.shields.io/badge/license-MIT-blue)  ![Python](https://img.shields.io/badge/python-3.9%2B-blue)

> **Uma suíte Python de código aberto para aplicar a Análise Espectral de Holo-Hilbert (HHSA) a dados neurofisiológicos complexos — EEG, ECoG, LFP.**

---

## Visão Geral

A **PyHHSA** é projetada para eliminar as distorções introduzidas por métodos espectrais clássicos baseados em expansões aditivas lineares (Fourier, Wavelets), permitindo a **quantificação direta de interações multiplicativas cross-scale coupling** no cérebro.

Focada em sinais estritamente **não-lineares e não-estacionários**, a biblioteca foi desenvolvida com foco em **Brain Decoding** e **Machine Learning**, extraindo diretamente matrizes de características de modulação de amplitude e frequência (AM–FM) prontas para classificadores como **SVMs e Redes Neurais**.

---

## O Problema Científico

Métodos espectrais clássicos sofrem de um defeito estrutural ao lidar com sistemas complexos: tratam processos multiplicativos (dinâmica inter-modo) com expansões puramente aditivas. Isso gera:

- *Ruído de banda larga fantasma*
- Espalhamento da energia de modulação
- Perda da assinatura de osciladores neurais acoplados

A **HHSA** resolve esse problema por meio de uma **Decomposição Empírica de Modos (EMD) Aninhada**:

1. **Camada FM:** Decompõe-se o sinal nas portadoras de frequência da rede neural.
2. **Camada AM:** Extraem-se os envelopes de amplitude de cada portadora e submete-os a uma nova EMD para isolar as moduladoras lentas.
3. **Espectro HHS (HHSM):** Constroi-se um espectro genuinamente multidimensional `(FM, AM, Potência)` que preserva a coerência de fase e a natureza multiplicativa dos dados.

---

## Arquitetura Modular

A suíte é dividida em **7 módulos independentes**, seguindo a ordem causal de processamento de um sinal neural:

| Módulo | Responsabilidade Principal | Destaques Científicos |
|--------|----------------------------|-----------------------|
| `io` | Entrada, Saída e Pré-processamento | Leitura via MNE-Python (EDF, NWB). Filtros FIR zero-phase para evitar distorção da Transformada de Hilbert |
| `decomposition` | Algoritmos de Decomposição Adaptativa | ICEEMDAN (1ª camada) e Masking-EMD (2ª camada) para mitigação severa de mode-mixing. Suporte a VMD |
| `hilbert` | Frequências e Amplitudes Instantâneos | *Normalized Hilbert Transform* (NHT) empírica, cálculo de IF por gradiente e extração de envelopes por Spline Natural |
| `hhsa` | Núcleo Matemático | Nested EMD, orquestração da decomposição de 2 camadas, mapeamento AM–FM tridimensional e extração do Índice de Não-Linearidade (Nlin) |
| `coupling` | Acoplamento de Redes Cross-Scale | PAC data-driven sem filtros pré-definidos — MVL, *Modulation Index* e Coerência Espectral |
| `visualization` | Inspeção Interativa e Gráficos | Gráficos 3D interativos da matriz HHSA (Plotly), Topomaps cerebrais de conectividade (MNE) e comparações Fourier vs. HHSA |
| `features` | Extração para Brain Decoding | Recorte de ROI (Potência total, Mean Frequencies `cf_FM`, `cf_AM`) e exportação automatizada para `X, y` compatíveis com Scikit-Learn |

---

## Instalação e Dependências

Para rodar a PyHHSA, garanta que seu ambiente possui as seguintes bibliotecas científicas:

```bash
pip install numpy scipy pandas scikit-learn matplotlib plotly
pip install mne EMD-signal vmdpy
```

Ou clone diretamente:

```bash
git clone https://github.com/labneoc/PyHHSA.git
cd PyHHSA
```

---

## Exemplo Rápido de Uso (Pipeline Linear)

```python
import numpy as np

# 1. Importação Modular
from pyhhsa.decomposition.selector import DecompositionSelector
from pyhhsa.hilbert.envelope import EnvelopeExtractor
from pyhhsa.hhsa.nestedemd import NestedEMD
from pyhhsa.hhsa.hhsbuilder import HoloHilbertSpectrum
from pyhhsa.coupling.pac import HHSAPAC
from pyhhsa.features.roipower import ROIPower

# 2. Dados — Sintéticos ou Reais (EEG)
fs = 1000  # Hz
time = np.linspace(0, 2, fs * 2)
signal = np.sin(2 * np.pi * 40 * time) * (1 + 0.5 * np.sin(2 * np.pi * 4 * time))

# 3. Configurando os Decompositores
# ICEEMDAN para FM, Masking para AM
fm_decomp = DecompositionSelector.get_decomposer(
    type='ICEEMDAN', n_trials=20, max_imfs=5
)

am_decomp = DecompositionSelector.get_decomposer(
    type='MASKING', mask_freq=50, mask_amp=1.0, fs=fs
)

env_ext = EnvelopeExtractor()

# 4. Processamento Central — Core HHSA
hhsa_engine = NestedEMD(fm_decomp, am_decomp, env_ext)
hhsa_results = hhsa_engine.decompose(signal)

# 5. Construção Espectral
hhs_matrix = HoloHilbertSpectrum(
    fm_bins=100, am_bins=50
).build_spectrum(
    if_fm=hhsa_results.if_fm,
    if_am=hhsa_results.if_am,
    ia_am=hhsa_results.ia_am
)

# 6. Extração de Features para Brain Decoding
power_beta_delta = ROIPower.extract_power_in_band(
    hhs_matrix, fm_edges, am_edges,
    fm_band=[13, 30],    # Beta
    am_band=[0, 4]       # Delta
)
print(f"Potência da modulação Delta-Beta extraída: {power_beta_delta:.3f}")
```

---

## Matriz Espectral HHS (HHSM)

A saída central da PyHHSA é uma matriz 3D `(FM, AM, Potência)` que representa:

- **Eixo FM (Frequência Instantânea):** Portadoras neurais (e.g., Alfa 8–13 Hz, Beta 13–30 Hz)
- **Eixo AM (Frequência de Modulação):** Moduladoras lentas (e.g., Theta 4–8 Hz, Delta 0–4 Hz)
- **Intensidade (Potência):** Força do acoplamento cross-scale em cada par de bandas

Isso difere radicalmente do espectrograma clássico, pois:

| Critério | Fourier/Wavelet | Holo-Hilbert (HHSA) |
|----------|----------------|--------------------|
| Modelo do sinal | Aditivo linear | Multiplicativo não-linear |
| Resolução temporal | Fixa (janela) | Adaptativa (EMD) |
| Acoplamento AM–FM | Indireto / ambíguo | Direto e mensurado |
| Base | Pré-definida (senoides) | Data-driven (modos IMFs) |
| Phase locking | Requer pós-processamento | Nativo (NHT) |
| Ruído fantasma | Alto em sinais não-estacionários | Eliminado pela EMD aninhada |

---

## Funcionalidades Avançadas

### Brain Decoding Suportado

- Extração de ROI por bandas clássicas: **Delta (0–4 Hz)**, **Theta (4–8 Hz)**, **Alpha (8–13 Hz)**, **Beta (13–30 Hz)**, **Gamma (30–100 Hz)**
- *Mean Frequency* dos componentes: `cf_FM`, `cf_AM`
- Exportação direta para `scikit-learn` (`X`, `y`)
- Pipeline completo para **Classificação**, **Regressão** e **Dimensionality Reduction**

### Outras Funcionalidades

- **PAC (Phase-Amplitude Coupling):** `Modulation Index`, `Mean Vector Length (MVL)` e `Coerência Espectral`
- **Índice de Não-Linearidade (Nlin):** Quantificação da assimetria espectral
- **Topomaps Cerebrais:** Visualização de conectividade multiescala com MNE-Python
- **Comparação Fourier vs HHSA:** Gráficos lado a lado para validação qualitativa

---

## Contribuição

Contribuições são bem-vindas! Siga os passos abaixo:

1. Fork este repositório
2. Crie um branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Faça commit das suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para o branch (`git push origin feature/nova-funcionalidade`)
5. Crie um Pull Request

Para reportar bugs ou sugerir melhorias, abra uma **issue** direta no repositório.

---

## Referências Científicas Principais

1. **Huang, N. E., et al. (2016).** *On Holo-Hilbert spectral analysis: a full informational spectral representation for nonlinear and non-stationary data.* Philosophical Transactions of the Royal Society A.
2. **Hsu, A. L., et al. (2018).** *Analyses of EEG Oscillatory Activities Using Holo-Hilbert Transform.* IEEE Transactions on Neural Systems and Rehabilitation Engineering.
3. **Torres, M. E., et al. (2011).** *A complete ensemble empirical mode decomposition with adaptive noise.* ICASSP.
4. **Colominas, M. A., et al. (2014).** *Improved complete ensemble EMD: A suitable tool for biomedical signal processing.* Biomedical Signal Processing and Control.

---

## Autores e Afilição

- **Prof. Dr. Bruno Duarte Gomes** — Faculdade de Biotecnologia, ICB, UFPA
- **Laboratório de Neurofisiologia** — Eduardo Oswaldo Cruz, ICB, UFPA
- **Laboratório de Simulação e Biologia Computacional** — CCAD, UFPA

---

## Licença

Este projeto é disponibilizado sob a **Licença MIT**.

Para fins acadêmicos, favor referenciar este repositório e os autores fundamentais ao utilizá-lo em publicações de **decodificação neural e neurodinâmica de sistemas complexos**.

---

<div align="center">

**LNEOC — Lab of Computational Neuroscience @ UFPA**  
*Desenvolvendo métodos computacionais avançados para a neurociência brasileira.*

</div>
