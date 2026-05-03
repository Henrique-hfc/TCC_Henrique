import numpy as np
from framework import file_m2k, file_civa
from scipy.signal import envelope
import matplotlib.pyplot as plt
from scipy.io import loadmat

sim_matlab = loadmat('./DadosMultiReflex.mat')
ascan_matlab = sim_matlab['ptFurosProf'][0, 0]['AscanValues'][:, 14].astype(np.float64)
matlab_envelope = envelope(ascan_matlab)

sim_real = file_m2k.read("./Mono_5MHz_ferro.m2k", 5, 0.5, 'Gaussian')
ascan_real = (sim_real.ascan_data[900:,0,0,0])

sim_civa = file_civa.read("./mono_aco_linear_2462_r52.civa")
ascan_civa = sim_civa.ascan_data_sum[:,0,0]
civa_envelope = envelope(ascan_civa)

real_envelope = envelope(ascan_real)

plt.figure(1)

plt.plot(ascan_matlab[:])
plt.title('Ensaio Real 5 MHz')

plt.figure(3)
real_abs = real_envelope[0,:]/np.max(real_envelope[0,:])
plt.plot(real_abs,'b', label='Ensaio Real ')
plt.title("AScan REAL ENVELOPE")
civa_abs = civa_envelope[0,1195:]/np.max(civa_envelope[0,:])
plt.plot(civa_abs,'r', label='Ensaio CIVA')
plt.title("AScan CIVA ENVELOPE")
plt.legend()

plt.show()

# np.save("Alum_real_cFonte.npy", ascan_real_n[:4100])
# np.save("Alum_sim_cFonte.npy", ascan_sim)

print("a")