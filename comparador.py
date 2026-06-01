import numpy as np
from framework import file_m2k, file_civa
from scipy.signal import envelope
import matplotlib.pyplot as plt
from scipy.io import loadmat

# --- Ensaio real ---
sim_real2 = file_m2k.read("./Mono_5MHz_ferro.m2k", 5, 0.5, 'Gaussian')
ascan_real_env = envelope(sim_real2.ascan_data[300:, 0, 0, 0])
real_envelope_n = ascan_real_env[0, :] / np.max(ascan_real_env[0, :])

# --- Simulação viscoelástica ---
sim_visco = np.load("result_sim_Qp5MHz.npy")
ascan_visco = envelope(sim_visco[750:, 0]) # ajuste inicio da fonte 250 amostras
visco_envelope_n = ascan_visco[0, :] / np.max(ascan_visco[0, :])

# --- Simulação viscoelástica Qp media ---
sim_media = np.load("result_sim_media145.npy") 
ascan_media = envelope(sim_media[750:, 0])
visco_media_envelope_n = ascan_media[0, :] / np.max(ascan_media[0, :])

# --- Simulação viscoelástica Qp media e f0 5 MHz ---
sim_media_nn = np.load("result_sim_media145_5MHz.npy") 
ascan_media_nn = envelope(sim_media_nn[750:, 0])
visco_media_envelope_nn = ascan_media_nn[0, :] / np.max(ascan_media_nn[0, :])

# --- Simulação viscoelástica Qp em 4.7 MHz e 3 Zeners ---
sim_3z = np.load("result_4p7_3zn.npy") 
ascan_3z = envelope(sim_3z[750:, 0])
visco_3z_envelope = ascan_3z[0, :] / np.max(ascan_3z[0, :])

# --- Simulação viscoelástica 2D Qp em 4.7 MHz e 3 Zeners ---
sim_2d_3z = np.load("result_4p7_3zn_2D.npy")
ascan_2d_3z = envelope(sim_2d_3z[750:, 0])
visco_2d_3z_envelope = ascan_2d_3z[0, :] / np.max(ascan_2d_3z[0, :])

# --- Simulação viscoelástica Qp em 5.4 MHz e 3 Zeners ---
sim_54 = np.load("result_5p4MHz.npy") 
ascan_54 = envelope(sim_54[750:, 0])
visco_54_envelope = ascan_54[0, :] / np.max(ascan_54[0, :])

# --- Simulação viscoelástica Qp em 4 MHz e 3 Zeners ---
sim_4 = np.load("result_4MHz.npy") 
ascan_4 = envelope(sim_4[750:, 0])
visco_4_envelope = ascan_4[0, :] / np.max(ascan_4   [0, :])

# --- Vetores de tempo (em µs) ---
dt_real = 8e-3    # 8 ns = 0,008 µs
dt_visco = 4e-3   # 4 ns = 0,004 µs

t_real = np.arange(len(real_envelope_n)) * dt_real
t_visco = np.arange(len(visco_envelope_n)) * dt_visco

# --- Plot sobreposto na mesma escala ---
plt.figure(figsize=(10, 5))
plt.plot(t_real, real_envelope_n, 'b', label='Ensaio Real')
plt.plot(t_visco, visco_envelope_n, 'r', label='Simulação Visco Qp em 5 Mhz')
plt.plot(t_visco, visco_media_envelope_n, 'g', label='Simulação Visco Qp Média')
plt.plot(t_visco, visco_media_envelope_nn, 'm', label='Simulação Visco Qp Média 5 MHz')
plt.plot(t_visco, visco_3z_envelope, 'c', label='Simulação Visco Qp 4.7 MHz, 3 Zeners')
plt.plot(t_visco, visco_2d_3z_envelope, 'y', label='Simulação Visco 2D Qp 4.7 MHz, 3 Zeners')
plt.plot(t_visco, visco_54_envelope, 'k', label='Simulação Visco Qp 5.4 MHz, 3 Zeners')
plt.plot(t_visco, visco_4_envelope, 'orange', label='Simulação Visco Qp 4 MHz, 3 Zeners')
plt.xlabel('Tempo (µs)')
plt.ylabel('Amplitude normalizada')
plt.legend()
plt.grid(True)

plt.figure(2)
plt.plot(t_real, real_envelope_n, 'b', label='Ensaio Real')
plt.plot(t_visco, visco_3z_envelope, 'c', label='Simulação Visco Qp 4.7 MHz, 3 Zeners')
plt.plot(t_visco, visco_2d_3z_envelope, 'y', label='Simulação Visco 2D Qp 4.7 MHz, 3 Zeners')
plt.xlabel('Tempo (µs)')
plt.ylabel('Amplitude normalizada')
plt.legend()
plt.grid(True)
plt.show()