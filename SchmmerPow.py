import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1
from framework import file_m2k


# =============================================================================
# 1. PARÂMETROS FÍSICOS E NUMÉRICOS
# =============================================================================
D      = 24.2e-3               # espessura da peça [m]
a      = (0.25 * 25.4e-3) / 2  # raio do transdutor [m]  (0.25 pol → m)
cp     = 5900.0                # velocidade da onda longitudinal no aço [m/s]
fs     = 125e6                 # taxa de amostragem [Hz]
eps_f  = 0.05                  # fração ε/max|F| para o filtro de Wiener
f_min, f_max = 2.4, 6.2      # banda útil de análise [MHz]
R_front      = 0.52       

# Janelas de cada eco
ECO1 = (900,  1250)
ECO2 = (1950, 2300)


# =============================================================================
# 2. CARREGAMENTO DO SINAL
# =============================================================================
ensaio = file_m2k.read("./Mono_5MHz_ferro.m2k", 5, 0.5, 'Gaussian')
signal = ensaio.ascan_data[:, 0, 0, 0].astype(np.float64)


# =============================================================================
# 3. EXTRAÇÃO E JANELAMENTO DOS ECOS
# =============================================================================
def extrair_janela(sig, sa, sb):
    sa, sb = int(sa), int(sb)
    seg    = sig[sa:sb]
    return seg * np.hanning(len(seg))

e1 = extrair_janela(signal, *ECO1)   # segmento janelado do eco 1
e2 = extrair_janela(signal, *ECO2)   # segmento janelado do eco 2


# =============================================================================
# 4. TRANSFORMADA DE FOURIER
# =============================================================================

NFFT  = 2048
V1    = np.fft.rfft(e1, n=NFFT)            # V₀ᵐ(ω) do eco 1
V2    = np.fft.rfft(e2, n=NFFT)            # V₀ᵐ(ω) do eco 2
freqs = np.fft.rfftfreq(NFFT, d=1.0 / fs)  # vetor de frequências [Hz]


# =============================================================================
# 5. CORREÇÃO DE DIFRAÇÃO — Schmerr Eq. (9.56)
# =============================================================================
#   ECO1 (n=1): D_eff = D    →  x₁ = k·a²/(2D)
#   ECO2 (n=2): D_eff = 2D   →  x₂ = k·a²/(4D)

def Dp(freqs, a, D_eff, cp):
    """
    Retorna
    -------
    Dp(x) = 1 − exp(i·x)·[J₀(x) − i·J₁(x)],  x = k·a²/(2·D_eff)
    """
    f = np.where(freqs == 0, 1e-10, freqs)          # guarda: evita x=0
    x = (2.0 * np.pi * f / cp) * (a**2 / (2.0 * D_eff))   # x = k·a²/(2D_eff)
    return 1.0 - np.exp(1j * x) * (j0(x) - 1j * j1(x))

dp1 = Dp(freqs, a, D,       cp)   # Dp para ECO1: D_eff = D
dp2 = Dp(freqs, a, 2.0 * D, cp)   # Dp para ECO2: D_eff = 2D


# =============================================================================
# 6. DECONVOLUÇÃO COM FILTRO DE WIENER — Schmerr Eq. (9.60) ADAPTADA
# =============================================================================
#
#   exp(−2·α·D) = |B|·|F| / (|F|² + ε²)             [Schmerr Eq. 9.60]


aF   = np.abs(V1) * np.abs(dp2) / (np.abs(dp1) + 1e-30)   # |F(ω)|
aB   = np.abs(V2)                                           # |B(ω)|
eps  = eps_f * aF.max()                                     # ε = 0.05·max|F|

# Eq. (9.60) adaptada — inclui 1/R_front pois R₂₁ não cancela nesta config.
e2aD = ((aB * aF) / (aF**2 + eps**2)) * (1.0 / R_front)


# =============================================================================
# 7. EXTRAÇÃO DO COEFICIENTE DE ATENUAÇÃO α(f) 
# =============================================================================

alpha_np_m = -np.log(np.clip(e2aD, 1e-30, None)) / (2.0 * D)   # [Np/m]


# =============================================================================
# 8. ISOLAMENTO DA BANDA E CONVERSÃO DE UNIDADES — Schmerr Eq. (9.6)
# =============================================================================
#
#   α_dB/l = 8.686 · α        sendo  8.686 = 20·log₁₀(e)
#   α [Np/m]  ──÷ 1000──►  α [Np/mm]  ──× 8.686──►  α [dB/mm]

CONV_NP_DB = 20.0 * np.log10(np.e)   # = 8.6859...  (fator exato da Eq. 9.6)

fMHz = freqs / 1e6
mask = (fMHz >= f_min) & (fMHz <= f_max)

f_band        = fMHz[mask]
alpha_band_db = (alpha_np_m[mask] / 1e3) * CONV_NP_DB   # [dB/mm]


# =============================================================================
# 9. AJUSTES DA CURVA α(f) — Schmerr Seção 9.2.2
# =============================================================================

# --- Grau 1 ---
coefs1     = np.polyfit(f_band, alpha_band_db, 1)
alpha_fit1 = np.polyval(coefs1, f_band)

# --- Grau 2 ---
coefs2     = np.polyfit(f_band, alpha_band_db, 2)
alpha_fit2 = np.polyval(coefs2, f_band)

# --- Lei de potência (somente pontos α > 0 — log indefinido para α ≤ 0) ---
mask_pos      = alpha_band_db > 0
log_f         = np.log(f_band[mask_pos])
log_alpha     = np.log(alpha_band_db[mask_pos])
n_pow, ln_a   = np.polyfit(log_f, log_alpha, 1)   # regressão log-log → [n, ln(a)]
a_pow         = np.exp(ln_a)                        # a = exp(ln_a)
alpha_fit_pow = a_pow * f_band**n_pow               # reconstrução no domínio linear


# =============================================================================
# 10. CONVERSÃO PARA O FORMATO DE POTÊNCIA
# =============================================================================
#   "Wave attenuation"              → α₀  [dB/mm]
#   "Power of the attenuation rate" → n   (adimensional)
#   "Wave frequency"                → f₀  [MHz]

f0_civa     = 5.0                          # frequência nominal do transdutor [MHz]
alpha0_civa = a_pow * (f0_civa ** n_pow)   # α₀ = a · f₀ⁿ  [dB/mm]


# =============================================================================
# 11. IMPRESSÃO DOS RESULTADOS
# =============================================================================
print("\n" + "=" * 65)
print(f"  ATENUAÇÃO — MODELO DE SCHMERR  |  Banda {f_min}–{f_max} MHz")
print("=" * 65)

print("\n[Grau 1]")
print(f"  alpha(f) = {coefs1[0]:.6e} · f  +  {coefs1[1]:.6e}   [dB/mm]")

print("\n[Grau 2]")
print(f"  alpha(f) = {coefs2[0]:.6e} · f²  +  {coefs2[1]:.6e} · f  +  {coefs2[2]:.6e}   [dB/mm]")

print("\n[Lei de Potência]")
print("\n[Power attenuation law]")
print(f"  Wave attenuation          : {alpha0_civa:.6e}  dB/mm")
print(f"  Power of attenuation rate : {n_pow:.4f}")
print(f"  Wave frequency            : {f0_civa:.1f}  MHz")
print("=" * 65 + "\n")


# =============================================================================
# 12. PLOTAGEM
# =============================================================================
plt.figure(figsize=(11, 5))
plt.plot(f_band, alpha_band_db, 'k-',  lw=1.5, alpha=0.6,
         label='Atenuação medida')
plt.plot(f_band, alpha_fit1,    'g--', lw=2.0,
         label=f'Grau 1: {coefs1[0]:.3e}·f + {coefs1[1]:.3e}')
plt.plot(f_band, alpha_fit2,    'r-',  lw=2.0,
         label=f'Grau 2: {coefs2[0]:.3e}·f² + {coefs2[1]:.3e}·f + {coefs2[2]:.3e}')
plt.plot(f_band, alpha_fit_pow, 'b:',  lw=2.5,
         label=f'Potência: {a_pow:.3e}·f^{n_pow:.3f}  (n={n_pow:.3f})')

plt.xlabel('Frequência (MHz)')
plt.ylabel(r'$\alpha$ (dB/mm)')
plt.title('Curva de Atenuação — Schmerr Cap. 9\n'
          'Ajustes: Grau 1, Grau 2 e Lei de Potência')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()