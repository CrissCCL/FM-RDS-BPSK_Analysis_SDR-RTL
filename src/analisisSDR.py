from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, resample_poly, welch, spectrogram

# ===============================
# PARÁMETROS SDR
# ===============================
STATION_HZ = 106.7e6 # <--------------------------- coloque aqui la estación de radio elegida
FS_RF      = 2.048e6
GAIN_DB    = 35
FS_AUDIO   = 200e3
SHIFT_HZ   = 0
N_SAMPLES  = 4*1024*1024

# ===============================
# CAPTURA DE MUESTRAS
# ===============================
sdr = RtlSdr()
sdr.sample_rate = FS_RF
sdr.center_freq = STATION_HZ + SHIFT_HZ
sdr.gain = GAIN_DB
sdr.rtl_agc = False
iq = sdr.read_samples(N_SAMPLES)
sdr.close()

# ===============================
# FILTRADO Y RESAMPLE
# ===============================
n = np.arange(len(iq))
iq_shifted = iq * np.exp(-1j*2*np.pi*SHIFT_HZ*n/FS_RF)

BW = 120e3
numtaps = 129
lp = firwin(numtaps, BW/(FS_RF/2))
iq_filt = lfilter(lp, 1.0, iq_shifted)

iq_limited = iq_filt / (np.abs(iq_filt) + 1e-6)
iq_bb = resample_poly(iq_limited, up=25, down=256)
fs = FS_AUDIO

# ===============================
# DEMODULACIÓN FM
# ===============================
fm = np.angle(iq_bb[1:] * np.conj(iq_bb[:-1]))

# ===============================
# ESPECTRO BASEBAND (PSD -Welch)
# ===============================
nperseg = 16384
f_psd, Pxx = welch(fm, fs=fs, nperseg=nperseg, noverlap=nperseg//2, return_onesided=True)
mask_psd = (f_psd >= 0) & (f_psd <= 60e3)

# ===============================
# FFT ÚNICA
# ===============================
fft_size = 65536
seg = fm[:fft_size] * np.hanning(fft_size)
F = np.fft.rfft(seg)
F_db = 20*np.log10(np.abs(F)+1e-12)
freqs_fft = np.fft.rfftfreq(fft_size, d=1/fs)
mask_fft = (freqs_fft >= 0) & (freqs_fft <= 60e3)

# ===============================
# ESPECTROGRAMA
# ===============================
f_spec, t_spec, Sxx = spectrogram(fm, fs=fs, nperseg=2048, noverlap=1024, mode='psd')
mask_spec = (f_spec >= 0) & (f_spec <= 60e3)
Sxx_dB = 10*np.log10(Sxx[mask_spec, :]+1e-12)

# ===============================
# PLOTEO EN SUBPLOTS
# ===============================
fig, axs = plt.subplots(3,1, figsize=(12,12))

# --- PSD baseband ---
axs[0].semilogy(f_psd[mask_psd]/1e3, Pxx[mask_psd], color='black')
axs[0].set_title("Espectro FM estéreo (Welch) 0–60 kHz")
axs[0].set_xlabel("Frecuencia (kHz)")
axs[0].set_ylabel("PSD")
axs[0].axvspan(0, 15, color='skyblue', alpha=0.3, label='L+R')
axs[0].axvspan(23, 53, color='lightgreen', alpha=0.3, label='L-R')
axs[0].axvline(19, color='orange', linestyle='--', linewidth=1.5, label='Piloto 19 kHz')
axs[0].axvline(38, color='red', linestyle=':', linewidth=1.2, label='Subportadora 38 kHz')
axs[0].axvline(57, color='violet', linestyle='--', linewidth=1.5, label='RDS 57 kHz')
axs[0].legend(); axs[0].grid(True)
# --- FFT única ---
axs[1].plot(freqs_fft[mask_fft]/1e3, F_db[mask_fft], color='black')
axs[1].set_title("Espectro FM estéreo (FFT) 0–60 kHz")
axs[1].set_xlabel("Frecuencia (kHz)")
axs[1].set_ylabel("Magnitud (dB)")
axs[1].axvspan(0, 15, color='skyblue', alpha=0.3)
axs[1].axvspan(23, 53, color='lightgreen', alpha=0.3)
axs[1].axvline(19, color='orange', linestyle='--', linewidth=1.5)
axs[1].axvline(38, color='red', linestyle=':', linewidth=1.2)
axs[1].axvline(57, color='violet', linestyle='--', linewidth=1.5)
axs[1].grid(True)
# --- Espectrograma ---
im = axs[2].imshow(Sxx_dB, aspect='auto', extent=[t_spec[0], t_spec[-1], f_spec[mask_spec][0]/1e3, f_spec[mask_spec][-1]/1e3], origin='lower', cmap='inferno')
axs[2].set_title("Espectrograma FM estéreo 0–60 kHz")
axs[2].set_xlabel("Tiempo (s)")
axs[2].set_ylabel("Frecuencia (kHz)")
fig.colorbar(im, ax=axs[2], label='PSD [dB]')

plt.tight_layout()
plt.show()
