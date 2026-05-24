import numpy as np
import matplotlib.pyplot as plt
from framework import file_m2k
from scipy.special import j0, j1


# =============================================================================
# PARÂMETROS
# =============================================================================

# Transdutor
a      = (0.25 * 25.4e-3) / 2   # raio [m]
f0_MHz = 5.0
BW     = 0.5
fs     = 125e6                   
eps_f  = 0.05                   

# Meio 1: água
D1   = 20e-3      
rho1 = 1000.0     
cp1  = 1480.0     
Z1   = rho1 * cp1

# Meio 2: aço carbono
D2   = 24.2e-3    
rho2 = 7800.0    
cp2  = 5900.0     
Z2   = rho2 * cp2

# Coeficientes de reflexão e transmissão — Schmerr Eq. (6.4)
R12 = (Z2 - Z1) / (Z2 + Z1)   # reflexão água→aço
T12 = 2 * Z2    / (Z1 + Z2)   # transmissão água→aço
R21 = (Z1 - Z2) / (Z1 + Z2)   # reflexão aço→água 
T21 = 2 * Z1    / (Z1 + Z2)   # transmissão aço→água

# Banda de análise [MHz]
f_min, f_max = 2.4, 6

# Janelas dos ecos em passos
ECO1 = (1150, 1400)   
ECO2 = (2200, 2450)   


# =============================================================================
# CARREGAMENTO
# =============================================================================

ensaio = file_m2k.read("./Mono_5MHz_imersão.m2k", f0_MHz, BW, 'Gaussian')
signal = ensaio.ascan_data[:, 0, 0, 0].astype(np.float64)


# =============================================================================
# EXTRAÇÃO E JANELAMENTO DOS ECOS
# =============================================================================

seg1 = signal[ECO1[0]:ECO1[1]]
e1   = seg1 * np.hanning(len(seg1))

seg2 = signal[ECO2[0]:ECO2[1]]
e2   = seg2 * np.hanning(len(seg2))


# =============================================================================
# TRANSFORMADA DE FOURIER
# =============================================================================

NFFT  = int(2 ** np.ceil(np.log2(len(signal) * 2)))
V1    = np.fft.rfft(e1, n=NFFT)
V2    = np.fft.rfft(e2, n=NFFT)
freqs = np.fft.rfftfreq(NFFT, d=1.0 / fs)


# =============================================================================
# COEFICIENTE  DE DIFRAÇÃO
# =============================================================================

D_eff1 = D1 + (cp1 / cp2) * D2
D_eff2 = D1 + (cp1 / cp2) * 2 * D2

f_safe = np.where(freqs == 0, 1e-10, freqs)

x1  = (2.0 * np.pi * f_safe / cp1) * (a**2 / (2.0 * D_eff1))
dp1 = 1.0 - np.exp(1j * x1) * (j0(x1) - 1j * j1(x1))

x2  = (2.0 * np.pi * f_safe / cp1) * (a**2 / (2.0 * D_eff2))
dp2 = 1.0 - np.exp(1j * x2) * (j0(x2) - 1j * j1(x2))


# =============================================================================
# DECONVOLUÇÃO COM FILTRO DE WIENER — Schmerr Eq. (9.60)
# =============================================================================

aF  = np.abs(V1) * abs(T12) * 1.0 * abs(T21) * abs(R21) * np.abs(dp2) / \
      (abs(T12) * 1.0 * abs(T21) * (np.abs(dp1) + 1e-30))

aB  = np.abs(V2) * abs(T12) * 1.0 * abs(T21) / \
      (abs(T12) * 1.0 * abs(T21))

eps  = eps_f * aF.max()

e2aD = (aB * aF) / (aF**2 + eps**2)


# =============================================================================
# COEFICIENTE DE ATENUAÇÃO α(f)
# =============================================================================

alpha_np_m = -np.log(e2aD) / (2.0 * D2)   # [Np/m]



# =============================================================================
# 8. BANDA E CONVERSÃO DE UNIDADES — Schmerr Eq. (9.6)
# =============================================================================

fMHz = freqs / 1e6
mask = (fMHz >= f_min) & (fMHz <= f_max)

f_band   = fMHz[mask]
alpha_np_mm = alpha_np_m[mask] / 1e3            # [Np/mm]
alpha_db = (alpha_np_m[mask] / 1e3) * 8.686   # [dB/mm]

np.save("alpha_Np_m.npy",  alpha_np_m[mask])   # [Np/m]
np.save("alpha_Np_mm.npy", alpha_np_mm)         # [Np/mm]
np.save("alpha_dB_mm.npy", alpha_db)            # [dB/mm]
np.save("freq_band_MHz.npy", f_band)            # frequencias da banda [MHz]


# =============================================================================
# AJUSTES DA CURVA α(f) 
# =============================================================================

# Grau 1
coefs1     = np.polyfit(f_band, alpha_db, 1)
alpha_fit1 = np.polyval(coefs1, f_band)

# Grau 2
coefs2     = np.polyfit(f_band, alpha_db, 2)
alpha_fit2 = np.polyval(coefs2, f_band)

# Grau 3 
coefs3 = np.polyfit(f_band, alpha_db, 3)
alpha_fit3 = np.polyval(coefs3, f_band)

# Grau 5 
coefs5 = np.polyfit(f_band, alpha_db, 5)
alpha_fit5 = np.polyval(coefs5, f_band)

# =============================================================================
# RESULTADOS
# =============================================================================

print("\n[Grau 1]")
print(f"  α(f) = {coefs1[0]:.6e}·f + {coefs1[1]:.6e}   [dB/mm]")
print("\n[Grau 2]")
print(f"  α(f) = {coefs2[0]:.6e}·f² + {coefs2[1]:.6e}·f + {coefs2[2]:.6e}   [dB/mm]")
print("\n[Grau 3]")
print(f"  α(f) = {coefs3[0]:.6e}·f³ + {coefs3[1]:.6e}·f² + {coefs3[2]:.6e}·f + {coefs3[3]:.6e}   [dB/mm]")


# =============================================================================
# 12. PLOTAGEM
# =============================================================================

plt.figure(figsize=(11, 5))
plt.plot(f_band, alpha_np_mm,       'k-',  lw=1.5, alpha=0.6, label='Medido')

plt.xlabel('Frequência (MHz)')
plt.ylabel('α (Np/mm)')
plt.title(f'Atenuação Ultrassônica ')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()