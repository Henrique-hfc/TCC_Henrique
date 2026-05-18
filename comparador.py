import numpy as np
from framework import file_m2k, file_civa
from scipy.signal import envelope
import matplotlib.pyplot as plt
from scipy.io import loadmat

sim_real2 = file_m2k.read("./Mono_5MHz_imersão.m2k", 5, 0.5, 'Gaussian')    
ascan_real_n = (sim_real2.ascan_data[:,0,0,0])
real_envelope_n = envelope(ascan_real_n)

sim_civa = file_civa.read("./mono_aco_imersao_2p46g3.civa")
ascan_civa = sim_civa.ascan_data_sum[:,0,0]
civa_envelope = envelope(ascan_civa)

periodo_us = 60  # µs

n_real = len(ascan_real_n)          # 7500 amostras
n_civa = len(ascan_civa)            # 9599 amostras

t_real = np.linspace(0, periodo_us, n_real)   # µs
t_civa = np.linspace(0, periodo_us, n_civa)   # µs


plt.figure(1)
plt.plot(t_real, real_envelope_n[0,:] / np.max(real_envelope_n[0,:]), 'b', label='Ensaio Real')
plt.xlabel('Tempo (µs)')
plt.ylabel('Amplitude normalizada')
plt.title('Ensaio Real Env 5 MHz')
plt.legend()

plt.figure(2)
plt.plot(t_real, ascan_real_n[:])
plt.xlabel('Tempo (µs)')
plt.ylabel('Amplitude')
plt.title('Ensaio Real 5 MHz Imersão')

civa_abs = civa_envelope[0, 4000:] / np.max(civa_envelope[0, :])
t_civa_slice = t_civa[4000:]        # mesmo recorte no eixo de tempo

plt.figure(3)
plt.plot(t_civa_slice, civa_abs, 'r', label='Ensaio CIVA')
plt.xlabel('Tempo (µs)')
plt.ylabel('Amplitude normalizada')
plt.title('AScan CIVA ENVELOPE')
plt.legend()

plt.show()
print("a")