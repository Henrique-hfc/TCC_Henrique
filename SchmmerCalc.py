import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1
from framework import file_m2k

# ==========================================
# 1. Parâmetros Físicos e Numéricos
# ==========================================
D      = 24.2e-3       # espessura (m)
a      = (0.25 * 25.4e-3) / 2   # raio transdutor (m)
cp     = 5940.0        # velocidade P no aço (m/s)
fs     = 125e6         # taxa de amostragem
eps_f  = 0.05
f_min, f_max = 2.5, 6.5   # banda útil em MHz

# Coeficiente de reflexão da interface Frontal (aço -> acoplante -> transdutor).
R_front = 0.85 

# Gates em passos — eco 1 e eco 2 de fundo
ECO1 = (900, 1250)
ECO2 = (1950, 2300)

# ==========================================
# 2. Carregamento do Sinal
# ==========================================
ensaio = file_m2k.read("./Mono_5MHz_ferro.m2k", 5, 0.5, 'Gaussian')
signal = ensaio.ascan_data[:, 0, 0, 0].astype(np.float64)

# ==========================================
# 3. Extração e Janelamento (Otimizado)
# ==========================================
def extrair_otimizado(sig, sa, sb):
    """
    Recorta estritamente a região de interesse e aplica a janela de Hann.
    Reduz drasticamente o custo computacional da FFT.
    """
    sa, sb = int(sa), int(sb)
    seg = sig[sa:sb]
    
    # A janela de Hanning agora tem o tamanho exato do recorte
    janela = np.hanning(len(seg))
    return seg * janela

e1_recortado = extrair_otimizado(signal, *ECO1)
e2_recortado = extrair_otimizado(signal, *ECO2)

# ==========================================
# 4. Transformada de Fourier (com Zero-Padding)
# ==========================================
NFFT  = 2048
V1    = np.fft.rfft(e1_recortado, n=NFFT)
V2    = np.fft.rfft(e2_recortado, n=NFFT)
freqs = np.fft.rfftfreq(NFFT, d=1.0/fs)

# ==========================================
# 5. Correção de Difração (Eq. 9.56)
# ==========================================
def Dp(freqs, a, D_eff, cp):
    f   = np.where(freqs == 0, 1e-10, freqs)
    arg = 2 * np.pi * f / cp * a**2 / (2 * D_eff)
    return 1.0 - np.exp(1j * arg) * (j0(arg) - 1j * j1(arg))

dp1 = Dp(freqs, a, D,     cp)
dp2 = Dp(freqs, a, 2.0*D, cp)

# ==========================================
# 6. Deconvolução e Filtro de Wiener
# ==========================================
F   = V1 * (np.abs(dp2) / (np.abs(dp1) + 1e-30))
B   = V2
aF  = np.abs(F)
aB  = np.abs(B)
eps = eps_f * aF.max()

# Implementação do Wiener estabilizando a razão e descontando a perda por reflexão
e2aD  = ((aB * aF) / (aF**2 + eps**2)) * (1.0 / R_front)

alpha = -np.log(np.clip(e2aD, 1e-30, None)) / (2.0 * D)   # Np/m

# ==========================================
# 7. Isolamento da Banda, Conversão e Ajuste
# ==========================================
fMHz = freqs / 1e6
mask = (fMHz >= f_min) & (fMHz <= f_max)

f_band = fMHz[mask]
alpha_np_m = alpha[mask]

# Conversão: Np/m -> Np/mm -> dB/mm
# Fator de conversão: 1 Np = 20 * log10(e) dB (aprox. 8.686 dB)
CONV_NP_DB = 20 * np.log10(np.e)
alpha_band_db = (alpha_np_m / 1000.0) * CONV_NP_DB

# Ajuste linear para extração da equação da curva de atenuação (agora em dB/mm)
coefs = np.polyfit(f_band, alpha_band_db, 1)
alpha_fit = np.polyval(coefs, f_band)

# ---> IMPRESSÃO NO CONSOLE <---
print("\n" + "="*60)
print(f"RESULTADO: EQUAÇÃO DA ATENUAÇÃO (Banda {f_min} a {f_max} MHz)")
print(f"alpha(f) = {coefs[0]:.6e} * f + {coefs[1]:.6e}  [dB/mm]")
print("Onde 'f' deve ser inserido em MHz para obter a atenuação em dB/mm.")
print("="*60 + "\n")

# Plotagem
plt.figure(figsize=(10, 5))
plt.plot(f_band, alpha_band_db, 'k-', lw=1.5, alpha=0.7, label='Atenuação ')
plt.plot(f_band, alpha_fit, 'r--', lw=2, label=f'Ajuste Linear')

plt.xlabel('Frequência (MHz)')
plt.ylabel('$\\alpha$ (dB/mm)')
plt.title('Curva de Atenuação Extraída (em dB/mm)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()