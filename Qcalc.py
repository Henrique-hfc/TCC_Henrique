import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

c_P = 5900.0      # Velocidade P (m/s)
c_S = 3230.0      # Velocidade S (m/s)
rho = 7850.0      # Densidade (kg/m³)
freq_MHz = np.load('freq_band_MHz.npy')
alpha_Np_mm = np.load('alpha_Np_mm.npy')

freq_Hz = freq_MHz * 1e6           # MHz → Hz
alpha_Np_m = alpha_Np_mm * 1000    # Np/mm → Np/m

# Q = π/αλ = k/2α
# c = w/k = 2πf / k → k = 2πf / c
# Q = πf / (αc)
Q_P = (np.pi * freq_Hz) / (alpha_Np_m * c_P)

# Q^-1 
Q_inv_medido = 1.0 / Q_P

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: α(f)
ax1 = axes[0, 0]
ax1.plot(freq_MHz, alpha_Np_mm, 'b-', linewidth=2)
ax1.set_xlabel('Frequência (MHz)', fontsize=12)
ax1.set_ylabel('α (Np/mm)', fontsize=12)
ax1.set_title('Coeficiente de Atenuação α(f)', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Q_P(f)
ax2 = axes[0, 1]
ax2.plot(freq_MHz, Q_P, 'b-', linewidth=2, label='Medido')
ax2.set_xlabel('Frequência (MHz)', fontsize=12)
ax2.set_ylabel('Q_P', fontsize=12)
ax2.set_title('Fator de Qualidade Q_P(f)', fontsize=12)
ax2.grid(True, alpha=0.3)

# Plot 3: Q^(-1)(f)
ax3 = axes[1, 0]
ax3.plot(freq_MHz, Q_inv_medido, 'b-', linewidth=2, label='Medido')
ax3.set_xlabel('Frequência (MHz)', fontsize=12)
ax3.set_ylabel('Q_P⁻¹', fontsize=12)
ax3.set_title('Inverso do Fator de Qualidade Q_P⁻¹(f)', fontsize=12)
ax3.grid(True, alpha=0.3)

plt.show()
