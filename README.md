# 🧠 PyHHSA: Holo-Hilbert Spectral Analysis Suite for Electrophysiology

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Neuroscience](https://img.shields.io/badge/domain-Neuroscience-purple.svg)

A **PyHHSA** é uma suíte Python de código aberto, projetada para aplicar o estado da arte da **Análise Espectral de Holo-Hilbert (HHSA)** a dados neurofisiológicos complexos (EEG, ECoG, LFP). Focada em sinais estritamente não-lineares e não-estacionários, a biblioteca elimina as distorções introduzidas por métodos baseados em expansões aditivas lineares (Fourier e Wavelets), permitindo a quantificação direta de interações multiplicativas (*cross-scale coupling*) no cérebro.

Desenvolvido com foco em **Brain Decoding e Machine Learning**, o pipeline extrai diretamente matrizes de características de modulação de amplitude e frequência (AM-FM) prontas para classificadores como SVMs e Redes Neurais.

---

## 🔬 O Problema Científico e a Solução Holo-Hilbert

Métodos espectrais clássicos sofrem de um defeito estrutural ao lidar com sistemas complexos: eles forçam processos multiplicativos (dinâmica inter-modo) em expansões puramente aditivas. Isso gera "ruído de banda larga" fantasma, espalhando a energia da modulação e destruindo a assinatura de osciladores neurais acoplados.

A **HHSA** resolve esse problema por meio de uma **Decomposição Empírica de Modos (EMD) Aninhada**:
1. Decompõe-se o sinal nas **portadoras de frequência (FM)** da rede neural.
2. Extraem-se os envelopes de amplitude de cada portadora e submete-os a uma nova EMD para isolar as **moduladoras lentas (AM)**.
3. Constrói-se um espectro genuinamente multidimensional $HH(\omega_{FM}, \omega_{AM}, P)$ que preserva a coerência de fase e a natureza multiplicativa dos dados.

---

## 🏗️ Arquitetura Modular

A suíte é dividida em 7 módulos independentes, seguindo a ordem causal de processamento de um sinal neural real:

| Módulo | Responsabilidade Principal | Destaques Científicos |
| :--- | :--- | :--- |
| 📦 **`io`** | Entrada, Saída e Pré-processamento | Leitura via `MNE-Python` (EDF, NWB). Filtros FIR *zero-phase* para evitar distorção da Transformada de Hilbert. |
| 🪚 **`decomposition`** | Algoritmos de Decomposição Adaptativa | Implementa **ICEEMDAN** (1ª camada) e **Masking-EMD** (2ª camada) para mitigação severa de *mode-mixing*. Suporte a VMD. |
| 🌊 **`hilbert`** | Frequências e Amplitudes Instantâneas | *Normalized Hilbert Transform* (NHT) empírica, cálculo de IF por gradiente e extração de envelopes por *Spline Natural*. |
| 🧩 **`hhsa`** | Core Matemático (Nested EMD) | Orquestração da decomposição de 2 camadas, mapeamento AM-FM tridimensional e extração do **Índice de Não-Linearidade ($N_{in}$)**. |
| 🔗 **`coupling`** | Acoplamento de Redes (Cross-Scale) | Cálculo *data-driven* de Phase-Amplitude Coupling (PAC) sem filtros pré-definidos (MVL, Modulation Index) e Coerência Espectral. |
| 📊 **`visualization`** | Inspeção Interativa e Gráficos | Gráficos 3D interativos da matriz HHSA no `Plotly`, *Topomaps* cerebrais de conectividade com MNE e comparações Fourier vs. HHSA. |
| 🧠 **`features`** | Extração para Brain Decoding | Recorte de ROI ($P_{total}$), *Mean Frequencies* ($cf_{FM}$, $cf_{AM}$) e exportação automatizada para $X$ e $y$ compatíveis com `Scikit-Learn`. |

---

## 🚀 Instalação e Dependências

Para rodar a PyHHSA, garanta que seu ambiente possui as seguintes bibliotecas científicas:

```bash
pip install numpy scipy pandas scikit-learn matplotlib plotly
pip install mne EMD-signal vmdpy
```

Basta clonar o repositório e os pacotes estarão disponíveis para importação no seu script raiz:
```bash
git clone https://github.com/SeuUsuario/PyHHSA.git
cd PyHHSA
```

---

## 💻 Exemplo Rápido de Uso (Pipeline Linear)

Abaixo um exemplo minimalista simulando como os 7 módulos interagem no processamento de uma época de EEG:

```python
import numpy as np

# 1. Importação Modular
from pyhhsa.decomposition.selector import DecompositionSelector
from pyhhsa.hilbert.envelope import EnvelopeExtractor
from pyhhsa.hhsa.nested_emd import NestedEMD
from pyhhsa.hhsa.hhs_builder import HoloHilbertSpectrum
from pyhhsa.coupling.pac import HHSAPAC
from pyhhsa.features.roi_power import ROIPower

# 2. Dados Sintéticos ou Reais (EEG)
fs = 1000
time = np.linspace(0, 2, fs*2)
signal = np.sin(2 * np.pi * 40 * time) * (1 + 0.5 * np.sin(2 * np.pi * 4 * time))

# 3. Configuração dos Decompositores (ICEEMDAN para FM, Masking para AM)
fm_decomp = DecompositionSelector.get_decomposer('ICEEMDAN', trials=20, max_imfs=5)
am_decomp = DecompositionSelector.get_decomposer('MASKING', mask_freq=50, mask_amp=1.0, fs=fs)
env_ext = EnvelopeExtractor()

# 4. Processamento Central (Core HHSA)
hhsa_engine = NestedEMD(fm_decomp, am_decomp, env_ext)
hhsa_results = hhsa_engine.decompose(signal)

# 5. Construção Espectral
hhs_matrix = HoloHilbertSpectrum(fm_bins=100, am_bins=50).build_spectrum(
    if_fm=hhsa_results['if_fm'], 
    if_am=hhsa_results['if_am'], 
    ia_am=hhsa_results['ia_am']
)

# 6. Extração de Features (Brain Decoding)
power_beta_delta = ROIPower.extract_power_in_band(
    hhs_matrix, fm_edges, am_edges, fm_band=(13, 30), am_band=(0, 4)
)
print(f"Potência da modulação Delta-Beta extraída: {power_beta_delta:.3f}")
```

---

## 📖 Referências Científicas Principais

O arcabouço matemático desta biblioteca repousa sob os avanços da Teoria de Sistemas Complexos e Análise de Sinais Não-lineares:

1. **Huang, N. E., et al. (2016).** *On Holo-Hilbert spectral analysis: a full informational spectral representation for nonlinear and non-stationary data.* Philosophical Transactions of the Royal Society A.
2. **Hsu, A. L., et al. (2018).** *Analyses of EEG Oscillatory Activities Using Holo-Hilbert Transform.* IEEE Transactions on Neural Systems and Rehabilitation Engineering.
3. **Torres, M. E., et al. (2011).** *A complete ensemble empirical mode decomposition with adaptive noise.* ICASSP.
4. **Colominas, M. A., et al. (2014).** *Improved complete ensemble EMD: A suitable tool for biomedical signal processing.* Biomedical Signal Processing and Control.

---

## ✒️ Autoria e Licença

**Prof. Dr. Bruno Duarte Gomes**  
Faculdade de Biotecnologia, ICB, UFPA  
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB, UFPA  
Laboratório de Simulação e Biologia Computacional, CCAD, UFPA  

Este projeto é disponibilizado sob a **Licença MIT**. Para fins acadêmicos, por favor, referencie este repositório e os autores fundamentais ao utilizá-lo em suas publicações de decodificação neural e neurodinâmica de sistemas complexos.
