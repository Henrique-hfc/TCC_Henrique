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

# --- Simulação viscoelástica Qp em 4.7 MHz e 3 Zeners ---
sim_3z = np.load("result_4p7_3zn.npy") 
ascan_3z = envelope(sim_3z[750:, 0])
visco_3z_envelope = ascan_3z[0, :] / np.max(ascan_3z[0, :])

# --- Simulação viscoelástica 2D Qp em 4.7 MHz e 3 Zeners ---
sim_2d_3z = np.load("result_4p7_3zn_2D.npy")
ascan_2d_3z = envelope(sim_2d_3z[750:, 0])
visco_2d_3z_envelope = ascan_2d_3z[0, :] / np.max(ascan_2d_3z[0, :])

# --- Simulação viscoelástica média Q ---
sim_mediaQ = np.load("result_mediaQ_reg.npy") 
ascan_mediaQ = envelope(sim_mediaQ[750:, 0])
visco_mediaQ_envelope = ascan_mediaQ[0, :] / np.max(ascan_mediaQ[0, :])

# --- Simulação viscoelástica média alfa ---
sim_mediaA = np.load("result_media_reg.npy") 
ascan_mediaA = envelope(sim_mediaA[750:, 0])
visco_mediaA_envelope = ascan_mediaA[0, :] / np.max(ascan_mediaA[0, :])
# --- Simulação viscoelástica média Q ---
sim_mediaQ = np.load("result_mediaQ_reg.npy") 
ascan_mediaQ = envelope(sim_mediaQ[750:, 0])
visco_mediaQ_envelope = ascan_mediaQ[0, :] / np.max(ascan_mediaQ[0, :])

# --- Simulação viscoelástica média alfa ---
sim_mediaA = np.load("result_media_reg.npy") 
ascan_mediaA = envelope(sim_mediaA[750:, 0])
visco_mediaA_envelope = ascan_mediaA[0, :] / np.max(ascan_mediaA[0, :])

# --- Simulação viscoelástica considerando Q invidiual coef maior 1 ---
sim_Qin = np.load("result_maior1.npy") 
ascan_Qin = envelope(sim_Qin[750:, 0])
visco_Qin_envelope = ascan_Qin[0, :] / np.max(ascan_Qin[0, :])

# --- Simulação viscoelástica considerando Q invidiual coef maior 2 ---
sim_Qin2 = np.load("result_maior2.npy") 
ascan_Qin2 = envelope(sim_Qin2[750:, 0])
visco_Qin_envelope2 = ascan_Qin2[0, :] / np.max(ascan_Qin2[0, :])

# --- Simulação viscoelástica alternativa ---
sim_alt = np.load("result_media_alt.npy") 
ascan_alt = envelope(sim_alt[750:, 0])
visco_alt_envelope = ascan_alt[0, :] / np.max(ascan_alt[0, :])

# --- Simulação viscoelástica considerando Q invidiual coef maior 1 ---
sim_Qin = np.load("result_maior1.npy") 
ascan_Qin = envelope(sim_Qin[750:, 0])
visco_Qin_envelope = ascan_Qin[0, :] / np.max(ascan_Qin[0, :])

# --- Simulação viscoelástica considerando Q invidiual coef maior 2 ---
sim_Qin2 = np.load("result_maior2.npy") 
ascan_Qin2 = envelope(sim_Qin2[750:, 0])
visco_Qin_envelope2 = ascan_Qin2[0, :] / np.max(ascan_Qin2[0, :])

# --- Simulação viscoelástica alternativa ---
sim_alt = np.load("result_media_alt.npy") 
ascan_alt = envelope(sim_alt[750:, 0])
visco_alt_envelope = ascan_alt[0, :] / np.max(ascan_alt[0, :])


# --- Vetores de tempo (em µs) ---
dt_real = 8e-3    # 8 ns = 0,008 µs
dt_visco = 4e-3   # 4 ns = 0,004 µs

t_real = np.arange(len(real_envelope_n)) * dt_real
t_visco = np.arange(len(visco_envelope_n)) * dt_visco

# --- Plot sobreposto na mesma escala ---
plt.figure(figsize=(10, 5))
plt.plot(t_real, real_envelope_n, 'b', label='Ensaio Real')
plt.plot(t_visco, visco_3z_envelope, 'c', label='Simulação Visco Qp 4.7 MHz, 3 Zeners')
plt.plot(t_visco, visco_mediaQ_envelope, 'purple', label='Simulação Média Q')
plt.plot(t_visco, visco_mediaA_envelope, 'pink', label='Simulação Média Alfa')
plt.plot(t_visco, visco_Qin_envelope, 'orange', label='Simulação Q coef maior 1')
plt.plot(t_visco, visco_Qin_envelope2, 'red', label='Simulação Q coef maior 2')
plt.plot(t_visco, visco_mediaQ_envelope, 'purple', label='Simulação Média Q')
plt.plot(t_visco, visco_mediaA_envelope, 'pink', label='Simulação Média Alfa')
plt.plot(t_visco, visco_Qin_envelope, 'orange', label='Simulação Q coef maior 1')
plt.plot(t_visco, visco_Qin_envelope2, 'red', label='Simulação Q coef maior 2')
plt.xlabel('Tempo (µs)')
plt.ylabel('Amplitude normalizada')
plt.legend()
plt.grid(True)

plt.figure(2)
plt.plot(t_real, real_envelope_n, 'b', label='Ensaio Real')
plt.plot(t_visco, visco_alt_envelope, 'green', label='Simulação Usando Versao ALternativa')
plt.plot(t_visco, visco_3z_envelope, 'c', label='Simulação Visco Qp 4.7 MHz, 3 Zeners')
plt.xlabel('Tempo (µs)')
plt.ylabel('Amplitude normalizada')
plt.legend()
plt.grid(True)
plt.show()