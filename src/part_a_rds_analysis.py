import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, resample_poly, welch, hilbert, spectrogram


# ============================================================
# CONFIGURACIÓN DE EJECUCIÓN
# ============================================================
# Ajuste estos parámetros al inicio, igual que en la Parte B.
#
# Si CAPTURE_NEW = True:
#   captura desde RTL-SDR y luego analiza.
#
# Si CAPTURE_NEW = False:
#   analiza el archivo IQ_FILENAME ya existente.

STATION_MHZ = 98.9 # <-----coloque aqui la estacion de radio elegida
CAPTURE_SECONDS = 65.0
GAIN_DB = 30.0
IQ_FILENAME = "fm_rds_iq_capture.npz"

# Evita confundir capturas: si AUTO_IQ_FILENAME=True,
# el archivo se nombra según la frecuencia, por ejemplo:
#   fm_rds_iq_98_1MHz.npz
AUTO_IQ_FILENAME = True

# Si True, obliga a que el archivo cargado corresponda a STATION_MHZ.
# Si detecta que el NPZ es de otra frecuencia, detiene el programa.
STRICT_STATION_MATCH = True

CAPTURE_NEW = False
SAVEFIGS = True
FIGDIR = "figuras_rds_A"


# ============================================================
# PARÁMETROS DSP / RDS
# ============================================================

FS_RF_DEFAULT = 2.048e6
FS_MPX = 228e3
FS_RDS = 19e3

BW_FM = 120e3
RDS_FC = 57e3

RDS_BPF_LOW = 54.0e3
RDS_BPF_HIGH = 60.0e3
RDS_LPF_HZ = 3.8e3

NUMTAPS_SYNC = 2001

COSTAS_ALPHA = 0.030
COSTAS_BETA = 0.00025

# ------------------------------------------------------------
# Modo de búsqueda de RDS
# ------------------------------------------------------------
# AUTO:
#   modo genérico. No prioriza una emisora conocida. Explora una grilla
#   de sincronización para intentar decodificar cualquier emisora con RDS.
#
# QUICK_KNOWN:
#   modo rápido para una emisora previamente caracterizada. En esta guía
#   se conserva como ejemplo opcional, pero NO se usa por defecto.
#
# Recomendación para laboratorio con varias emisoras:
#   SEARCH_MODE = "AUTO"
SEARCH_MODE = "AUTO"   # "AUTO" o "QUICK_KNOWN"

# Candidatos rápidos opcionales para una emisora ya caracterizada.
# No se usan cuando SEARCH_MODE = "AUTO".
QUICK_KNOWN_CANDIDATES = [
    (16.000, 2.67),
    (15.980, 10.65),
    (15.980, 5.33),
    (16.020, 14.68),
]

# Grilla genérica.
# RDS tiene 19000/1187.5 = 16 muestras/bit en esta implementación.
# Se prueban pequeñas variaciones de sps y offsets dentro de un bit completo.
AUTO_SPS_VALUES = [15.96, 15.98, 16.00, 16.02, 16.04]
AUTO_OFFSET_DIVISIONS = 16

# Rutas de bajada de la subportadora RDS:
# - piloto_19k_al_cubo: usa el piloto estéreo de 19 kHz elevado al cubo.
# - oscilador_fijo_57k: usa un oscilador local ideal de 57 kHz.
#
# En algunas emisoras una ruta puede recuperar segmentos que la otra no.
RDS_ROUTES = ["piloto_19k_al_cubo", "oscilador_fijo_57k"]

# Si se completa el PS durante la búsqueda, se detiene para ahorrar tiempo.
STOP_WHEN_PS_COMPLETE_PART_A = True

MIN_GROUPS_FOR_VALID_RDS = 3



def build_corr_candidates():
    """
    Construye candidatos de sincronización para el correlador bifase.

    En modo AUTO se usa una grilla genérica de sps/offset. No se priorizan
    parámetros de una emisora particular, para que la experiencia sea aplicable
    a distintas estaciones.

    En modo QUICK_KNOWN se usan pocos candidatos ya caracterizados para acelerar
    una demostración con una emisora conocida.
    """
    mode = SEARCH_MODE.upper().strip()

    if mode == "QUICK_KNOWN":
        return list(QUICK_KNOWN_CANDIDATES)

    candidates = []
    for sps in AUTO_SPS_VALUES:
        for off in np.linspace(0, 16.0, AUTO_OFFSET_DIVISIONS, endpoint=False):
            candidates.append((float(sps), float(off)))

    return candidates


# ============================================================
# Utilidades DSP
# ============================================================

def tic():
    return time.perf_counter()


def toc(t0, msg):
    print(f"{msg}: {time.perf_counter() - t0:.2f} s")


def fir_filter_delay(x, taps):
    y = lfilter(taps, 1.0, x)
    delay = (len(taps) - 1) // 2
    return y[delay:]


def normalize_power(x):
    return x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)


def costas_loop_bpsk(x, alpha=COSTAS_ALPHA, beta=COSTAS_BETA):
    y_out = np.zeros_like(x, dtype=np.complex128)
    phase = 0.0
    freq = 0.0

    for i, sample in enumerate(x):
        y = sample * np.exp(-1j * phase)

        decision = 1.0 if np.real(y) >= 0 else -1.0
        error = decision * np.imag(y)

        freq += beta * error
        phase += freq + alpha * error
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi

        y_out[i] = y

    return y_out


def pca_rotate_bpsk(x):
    s = x.copy()
    center = np.median(np.real(s)) + 1j * np.median(np.imag(s))
    s = s - center

    X = np.column_stack((np.real(s), np.imag(s)))
    C = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    v = eigvecs[:, np.argmax(eigvals)]

    angle = np.arctan2(v[1], v[0])
    y = s * np.exp(-1j * angle)

    return y, angle


def estimate_residual_freq_bpsk_fft(x, fs, search_hz=300):
    x = x - np.mean(x)
    z = x ** 2

    N = len(z)
    nfft = 2 ** int(np.ceil(np.log2(N)))
    window = np.hanning(N)

    Z = np.fft.fftshift(np.fft.fft(z * window, n=nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs))

    mask = (f >= -2 * search_hz) & (f <= 2 * search_hz)
    f_search = f[mask]
    Z_search = Z[mask]

    f2_est = f_search[np.argmax(np.abs(Z_search))]
    f_est = f2_est / 2.0

    n = np.arange(len(x))
    y = x * np.exp(-1j * 2 * np.pi * f_est * n / fs)

    return y, f_est


# ============================================================
# RDS: validación básica de grupos
# ============================================================

def bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def rds_syndrome_lfsr(block_bits):
    # g(x)=x^10+x^8+x^7+x^5+x^4+x^3+1
    poly = 0x5B9
    reg = 0

    for b in block_bits:
        reg = (reg << 1) | int(b)
        if reg & (1 << 10):
            reg ^= poly

    return reg & 0x3FF


RDS_OFFSETS = {
    0x0FC: "A",
    0x198: "B",
    0x168: "C",
    0x350: "C'",
    0x1B4: "D",
}


def classify_block(block_bits):
    s = rds_syndrome_lfsr(block_bits)
    if s in RDS_OFFSETS:
        return RDS_OFFSETS[s], bits_to_int(block_bits[:16])
    return None, None


def find_rds_groups(bitstream, max_groups=300):
    """
    Busca grupos RDS A, B, C/C', D.
    Para sincronizar se exige A, B y D. C/C' se reporta si valida.
    """
    groups = []
    N = len(bitstream)

    for i in range(0, N - 104 + 1):
        bA = bitstream[i:i + 26]
        bB = bitstream[i + 26:i + 52]
        bC = bitstream[i + 52:i + 78]
        bD = bitstream[i + 78:i + 104]

        tA, dA = classify_block(bA)
        if tA != "A":
            continue

        tB, dB = classify_block(bB)
        if tB != "B":
            continue

        tD, dD = classify_block(bD)
        if tD != "D":
            continue

        tC, dC = classify_block(bC)

        groups.append({
            "pos": i,
            "A": dA,
            "B": dB,
            "C": dC if tC in ("C", "C'") else None,
            "D": dD,
            "C_type": tC if tC in ("C", "C'") else None,
        })

        if len(groups) >= max_groups:
            break

    return groups


def differential_decode(bits, init=0):
    out = np.zeros_like(bits)
    prev = init

    for i, b in enumerate(bits):
        out[i] = int(b) ^ int(prev)
        prev = int(b)

    return out.astype(np.uint8)


def decode_bitstream_variants(bits0):
    """
    Prueba las ambigüedades binarias habituales y devuelve:
      - best: mejor variante para graficar la extracción temporal;
      - results: todas las variantes evaluadas.

    Cambio importante respecto a versiones anteriores:
    Para consolidar PS no se usa solo el mejor modo. Se acumulan grupos
    válidos desde todas las variantes, igual que en la Parte B.
    """
    results = []
    raw_options = [
        ("raw", bits0),
        ("raw_inv", bits0 ^ 1),
    ]

    for raw_name, bits_raw in raw_options:
        streams = [
            (f"{raw_name}|sin diferencial", bits_raw),
            (f"{raw_name}|diferencial init=0", differential_decode(bits_raw, init=0)),
            (f"{raw_name}|diferencial init=1", differential_decode(bits_raw, init=1)),
        ]

        for mode, bs in streams:
            for final_inv in [0, 1]:
                b = bs ^ final_inv
                groups = find_rds_groups(b)

                results.append({
                    "mode": mode,
                    "final_inv": final_inv,
                    "bits": b,
                    "groups": groups,
                    "n_groups": len(groups),
                })

    results.sort(key=lambda r: r["n_groups"], reverse=True)
    return results[0], results


def printable_byte(v):
    return chr(v) if 32 <= v <= 126 else "?"


def vote_ps(groups):
    votes = {0: {}, 1: {}, 2: {}, 3: {}}

    for g in groups:
        B = g["B"]
        D = g["D"]
        group_type = (B >> 12) & 0xF

        if group_type != 0:
            continue

        segment = B & 0x03
        chars = printable_byte((D >> 8) & 0xFF) + printable_byte(D & 0xFF)
        votes[segment][chars] = votes[segment].get(chars, 0) + 1

    ps_pairs = ["??"] * 4
    seen = [False] * 4

    for seg in range(4):
        if votes[seg]:
            best_chars = max(votes[seg], key=votes[seg].get)
            ps_pairs[seg] = best_chars
            seen[seg] = True

    return "".join(ps_pairs), seen


def pi_counts(groups):
    d = {}
    for g in groups:
        d[g["A"]] = d.get(g["A"], 0) + 1
    return d


def consolidate_groups(groups):
    if not groups:
        return [], None

    counts = pi_counts(groups)
    pi_dom = max(counts, key=counts.get)

    unique = {}
    for g in groups:
        if g["A"] != pi_dom:
            continue
        key = (g["A"], g["B"], g["C"], g["D"], g["C_type"])
        unique[key] = g

    out = list(unique.values())
    out.sort(key=lambda g: g["pos"])
    return out, pi_dom


# ============================================================
# Correlador bifase para símbolos BPSK
# ============================================================

def biphase_metrics_complex(x_complex, sps_bit, bit_offset, n_int=8):
    """
    Filtro/correlador bifase por bit completo:
      métrica = promedio(primera mitad del bit) - promedio(segunda mitad)

    Retorna métrica compleja de decisión. Si RDS está bien extraído,
    esta métrica forma dos nubes BPSK sobre el eje real.
    """
    x = np.asarray(x_complex, dtype=np.complex128)
    n = np.arange(len(x), dtype=float)

    n_bits = int((len(x) - bit_offset - sps_bit) // sps_bit)
    if n_bits <= 10:
        return np.array([], dtype=np.complex128)

    frac_first = (np.arange(n_int) + 0.5) / (2 * n_int)
    frac_second = 0.5 + (np.arange(n_int) + 0.5) / (2 * n_int)

    starts = bit_offset + np.arange(n_bits)[:, None] * sps_bit
    idx1 = starts + frac_first[None, :] * sps_bit
    idx2 = starts + frac_second[None, :] * sps_bit

    xr = np.real(x)
    xi = np.imag(x)

    v1 = (
        np.interp(idx1.ravel(), n, xr).reshape(n_bits, n_int)
        + 1j * np.interp(idx1.ravel(), n, xi).reshape(n_bits, n_int)
    )
    v2 = (
        np.interp(idx2.ravel(), n, xr).reshape(n_bits, n_int)
        + 1j * np.interp(idx2.ravel(), n, xi).reshape(n_bits, n_int)
    )

    return np.mean(v1, axis=1) - np.mean(v2, axis=1)


def evaluate_candidate(rds_19k_corr, sps_bit, bit_offset):
    locked = costas_loop_bpsk(rds_19k_corr)
    locked = locked[int(0.25 * FS_RDS):]

    rot, angle = pca_rotate_bpsk(locked)
    metrics = biphase_metrics_complex(rot, sps_bit=sps_bit, bit_offset=bit_offset)

    if len(metrics) < 104:
        return None

    bits0 = (np.real(metrics) > np.median(np.real(metrics))).astype(np.uint8)
    best, all_results = decode_bitstream_variants(bits0)

    all_groups = []
    for r in all_results:
        for g in r["groups"]:
            gg = dict(g)
            gg["_candidate_sps"] = sps_bit
            gg["_candidate_offset"] = bit_offset
            gg["_mode"] = r["mode"]
            gg["_final_inv"] = r["final_inv"]
            all_groups.append(gg)

    # bits0 representa el estado de fase BPSK antes de probar inversión/diferencial.
    # Es lo más adecuado para visualizar los cambios de fase 0/pi.
    best["raw_phase_bits"] = bits0
    best["threshold"] = float(np.median(np.real(metrics)))
    best["metrics"] = metrics
    best["symbol_signal"] = rot
    best["sps_bit"] = sps_bit
    best["bit_offset"] = bit_offset
    best["angle"] = angle
    best["all_results"] = all_results
    best["all_groups"] = all_groups
    best["n_groups_all_variants"] = len(all_groups)

    return best



def plot_bit_extraction_time(symbol_signal, metrics, raw_phase_bits, decoded_bits,
                             sps_bit, bit_offset, bit_start, n_bits, threshold,
                             groups_best=None, savefigs=False, figdir="figuras_rds_A"):
    """
    Figura-resumen: muestra en una sola imagen cómo se extrae la
    trama binaria a partir de la señal RDS en el tiempo.
    """
    x = np.real(symbol_signal)
    sps = float(sps_bit)
    bit_offset = float(bit_offset)

    bit_start = int(max(0, bit_start))
    n_bits = int(min(n_bits, len(metrics) - bit_start, len(decoded_bits) - bit_start))
    if n_bits <= 6:
        return

    s0 = int(max(0, np.floor(bit_offset + bit_start * sps - 2 * sps)))
    s1 = int(min(len(x), np.ceil(bit_offset + (bit_start + n_bits) * sps + 2 * sps)))

    n = np.arange(s0, s1)
    t_ms = (n - (bit_offset + bit_start * sps)) / FS_RDS * 1e3
    xseg = x[s0:s1]

    x_center = xseg - np.median(xseg)
    x_scale = np.percentile(np.abs(x_center), 95) + 1e-12
    xnorm = x_center / x_scale

    first_avg = []
    second_avg = []
    metric_seg = []
    raw_seg = []
    dec_seg = []

    for kk in range(bit_start, bit_start + n_bits):
        st = bit_offset + kk * sps
        mid = st + sps / 2

        m = 10
        idx1 = st + (np.arange(m) + 0.5) * (sps / 2) / m
        idx2 = mid + (np.arange(m) + 0.5) * (sps / 2) / m

        vals1 = np.interp(idx1, np.arange(len(x)), x)
        vals2 = np.interp(idx2, np.arange(len(x)), x)

        a = np.mean(vals1)
        b = np.mean(vals2)
        first_avg.append(a)
        second_avg.append(b)
        metric_seg.append(metrics[kk])
        raw_seg.append(raw_phase_bits[kk])
        dec_seg.append(decoded_bits[kk])

    first_avg = np.asarray(first_avg)
    second_avg = np.asarray(second_avg)
    metric_seg = np.asarray(metric_seg)
    raw_seg = np.asarray(raw_seg)
    dec_seg = np.asarray(dec_seg)

    first_avg_norm = (first_avg - np.median(xseg)) / x_scale
    second_avg_norm = (second_avg - np.median(xseg)) / x_scale
    k_axis = np.arange(bit_start, bit_start + n_bits)

    fig, axs = plt.subplots(4, 1, figsize=(13.5, 10.5), sharex=False,
                            gridspec_kw={"height_ratios": [2.4, 1.25, 1.15, 1.2]})

    # Panel 1: señal y ventanas.
    axs[0].plot(t_ms, xnorm, linewidth=1.0, label="Señal RDS proyectada I(t)")
    axs[0].axhline(0, linewidth=0.8)

    for ii, kk in enumerate(k_axis):
        st = bit_offset + kk * sps
        mid = st + sps / 2
        en = st + sps

        st_ms = (st - (bit_offset + bit_start * sps)) / FS_RDS * 1e3
        mid_ms = (mid - (bit_offset + bit_start * sps)) / FS_RDS * 1e3
        en_ms = (en - (bit_offset + bit_start * sps)) / FS_RDS * 1e3
        cen1_ms = (st + sps / 4 - (bit_offset + bit_start * sps)) / FS_RDS * 1e3
        cen2_ms = (mid + sps / 4 - (bit_offset + bit_start * sps)) / FS_RDS * 1e3

        if ii % 2 == 0:
            axs[0].axvspan(st_ms, en_ms, alpha=0.08)
        axs[0].axvline(st_ms, linestyle=":", linewidth=0.7)
        axs[0].axvline(mid_ms, linestyle="--", linewidth=0.7)
        axs[0].plot(cen1_ms, first_avg_norm[ii], marker="o", markersize=5)
        axs[0].plot(cen2_ms, second_avg_norm[ii], marker="s", markersize=5)
        if ii < 8:
            axs[0].text((st_ms + en_ms) / 2, 1.14, str(dec_seg[ii]), ha="center", fontsize=8)

    end_ms = ((bit_offset + (bit_start + n_bits) * sps) - (bit_offset + bit_start * sps)) / FS_RDS * 1e3
    axs[0].set_xlim(-0.5, end_ms + 0.5)
    axs[0].set_ylim(-1.45, 1.45)
    axs[0].set_ylabel("Amplitud norm.")
    axs[0].set_title("Paso 1: señal RDS en el tiempo y división de cada bit en dos mitades")
    axs[0].grid(True)
    axs[0].legend(loc="lower right")

    # Panel 2: promedios por mitad.
    axs[1].plot(k_axis, first_avg_norm, marker="o", label="promedio mitad 1")
    axs[1].plot(k_axis, second_avg_norm, marker="s", label="promedio mitad 2")
    axs[1].axhline(0, linewidth=0.8)
    axs[1].set_ylabel("Promedio")
    axs[1].set_title("Paso 2: comparación de mitades del bit (correlador bifase)")
    axs[1].grid(True)
    axs[1].legend(loc="best")

    # Panel 3: métrica y fase.
    met_real = np.real(metric_seg)
    met_center = met_real - threshold
    colors = ["tab:blue" if v >= 0 else "tab:orange" for v in met_center]
    axs[2].bar(k_axis, met_center, color=colors, alpha=0.75, label=r"$m_k-\gamma$")
    axs[2].axhline(0, linestyle="--", linewidth=1.0)
    axs[2].set_ylabel(r"$m_k-\gamma$")
    axs[2].set_title("Paso 3: métrica de decisión; el signo indica la fase BPSK")
    axs[2].grid(True)
    axs[2].legend(loc="best")

    # Panel 4: fase y bits finales.
    phase_state = np.where(raw_seg > 0, 1.0, -1.0)
    axs[3].step(k_axis, phase_state, where="mid", linewidth=1.7, label="fase estimada")
    axs[3].step(k_axis, dec_seg - 0.5, where="mid", linewidth=1.4, label="bit decodificado")
    axs[3].set_yticks([-1, -0.5, 0.5, 1])
    axs[3].set_yticklabels(["fase π", "bit 0", "bit 1", "fase 0"])
    axs[3].set_xlabel("Índice de bit")
    axs[3].set_title("Paso 4: reconstrucción de la trama binaria")
    axs[3].grid(True)
    axs[3].legend(loc="best")

    if groups_best:
        first_group_pos = int(groups_best[0]["pos"])
        for ax in axs[1:]:
            for xg in [first_group_pos, first_group_pos + 26, first_group_pos + 52, first_group_pos + 78, first_group_pos + 104]:
                if bit_start <= xg <= bit_start + n_bits:
                    ax.axvline(xg, linestyle=":", linewidth=0.9)

    bit_txt = "".join(str(int(b)) for b in dec_seg[:min(len(dec_seg), 24)])
    fig.text(0.02, 0.01, f"Primeros bits en la ventana mostrada: {bit_txt}", family="monospace", fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    if savefigs:
        fig.savefig(Path(figdir) / "A4_resumen_extraccion_trama_rds.png", dpi=170)




def gather_valid_metric_indices(metrics, groups_cons, group_len=104, max_groups=8):
    """
    Devuelve índices de símbolos/metrics asociados a grupos RDS consolidados.

    Se usa solo para la figura de constelación, con el fin de no
    mezclar símbolos de grupos validados con ruido o regiones no sincronizadas.
    """
    if not groups_cons:
        return np.arange(min(len(metrics), 1500), dtype=int)

    idx = []
    used_groups = groups_cons[:max_groups] if len(groups_cons) > max_groups else groups_cons

    for g in used_groups:
        pos = int(g["pos"])
        a = max(0, pos)
        b = min(len(metrics), pos + group_len)
        idx.extend(range(a, b))

    if not idx:
        return np.arange(min(len(metrics), 1500), dtype=int)

    return np.array(sorted(set(idx)), dtype=int)


def rotate_bpsk_metrics_for_display(metrics, raw_phase_bits, idx_valid):
    """
    Rota la métrica compleja para alinear la BPSK con el eje real.

    Importante:
    - No estira los ejes I/Q.
    - No normaliza punto a punto.
    - Solo corrige una fase residual común.
    """
    m = np.asarray(metrics, dtype=np.complex128)
    raw = np.asarray(raw_phase_bits).astype(int)

    # Centrado robusto.
    m_center = m - (np.median(np.real(m)) + 1j * np.median(np.imag(m)))

    if idx_valid is None or len(idx_valid) == 0:
        idx_valid = np.arange(len(m_center), dtype=int)

    idx_valid = idx_valid[(idx_valid >= 0) & (idx_valid < len(m_center))]
    if len(idx_valid) == 0:
        idx_valid = np.arange(min(len(m_center), 1500), dtype=int)

    m_sel = m_center[idx_valid]
    mag_sel = np.abs(m_sel)

    if len(mag_sel) > 10:
        conf_thr = np.percentile(mag_sel, 45)
        conf_mask_sel = mag_sel > conf_thr
    else:
        conf_mask_sel = np.ones(len(mag_sel), dtype=bool)

    if np.count_nonzero(conf_mask_sel) < 12:
        conf_mask_sel = np.ones(len(mag_sel), dtype=bool)

    m_conf = m_sel[conf_mask_sel]

    # Para BPSK, elevar al cuadrado elimina la ambigüedad de signo.
    # La mitad de la fase promedio permite estimar la rotación residual común.
    if len(m_conf) > 0:
        theta = 0.5 * np.angle(np.mean((m_conf + 1e-12) ** 2))
    else:
        theta = 0.0

    m_rot = m_center * np.exp(-1j * theta)

    # Orientación: que raw_phase_bits=1 quede preferentemente a la derecha.
    idx_conf_global = idx_valid[conf_mask_sel]
    if len(idx_conf_global) > 0 and len(raw) == len(m_rot):
        pos_idx = idx_conf_global[raw[idx_conf_global] > 0]
        neg_idx = idx_conf_global[raw[idx_conf_global] == 0]

        if len(pos_idx) > 0 and len(neg_idx) > 0:
            if np.median(np.real(m_rot[pos_idx])) < np.median(np.real(m_rot[neg_idx])):
                m_rot = -m_rot

    # Escala común robusta para que los puntos ideales ±1 sean comparables.
    # Esto no deforma ejes: divide I y Q por el mismo factor.
    m_show = m_rot[idx_valid]
    scale = np.percentile(np.abs(np.real(m_show)), 75) + 1e-12 if len(m_show) else 1.0
    m_rot = m_rot / scale

    mag_rot_sel = np.abs(m_rot[idx_valid])
    if len(mag_rot_sel) > 10:
        conf_thr_rot = np.percentile(mag_rot_sel, 45)
    else:
        conf_thr_rot = 0.0

    conf_mask_global = np.zeros(len(m_rot), dtype=bool)
    conf_mask_global[idx_valid] = mag_rot_sel > conf_thr_rot

    return m_rot, conf_mask_global




def station_filename_from_mhz(station_mhz):
    txt = f"{station_mhz:.1f}".replace(".", "_")
    return f"fm_rds_iq_{txt}MHz.npz"


def resolve_iq_filename():
    if AUTO_IQ_FILENAME:
        return station_filename_from_mhz(STATION_MHZ)
    return IQ_FILENAME


def check_capture_matches_station(infile, station_mhz):
    """
    Verifica que el archivo NPZ corresponda a la frecuencia configurada.
    Evita analizar por error una captura antigua de otra emisora.
    """
    try:
        data = np.load(infile)
        if "station_hz" not in data.files:
            print("Advertencia: el archivo no contiene metadata station_hz.")
            return True

        file_mhz = float(data["station_hz"]) / 1e6
        diff_mhz = abs(file_mhz - station_mhz)

        if diff_mhz > 0.05:
            msg = (
                "\nERROR: El archivo de captura no corresponde a la emisora configurada.\n"
                f"  STATION_MHZ configurado: {station_mhz:.3f} MHz\n"
                f"  Archivo NPZ contiene:     {file_mhz:.3f} MHz\n\n"
                "Esto suele ocurrir cuando se cambia STATION_MHZ, pero se deja CAPTURE_NEW=False\n"
                "y se reutiliza una captura antigua.\n\n"
                "Solución:\n"
                "  1) Cambie CAPTURE_NEW=True para capturar la nueva emisora, o\n"
                "  2) cambie IQ_FILENAME al archivo correcto, o\n"
                "  3) active AUTO_IQ_FILENAME=True y genere una captura por frecuencia.\n"
            )
            if STRICT_STATION_MATCH:
                raise RuntimeError(msg)
            else:
                print(msg)
                return False

        return True

    except RuntimeError:
        raise
    except Exception as exc:
        print(f"Advertencia: no se pudo verificar metadata del archivo: {exc}")
        return True


# ============================================================
# Captura y procesamiento principal
# ============================================================

def capture_rtl_sdr(fc_mhz, seconds, gain, out, fs_rf=FS_RF_DEFAULT, block=262144):
    try:
        from rtlsdr import RtlSdr
    except Exception as exc:
        raise RuntimeError("No se pudo importar pyrtlsdr. Instale con: pip install pyrtlsdr") from exc

    n_total = int(seconds * fs_rf)
    print(f"Capturando {seconds:.1f} s en {fc_mhz:.3f} MHz")
    print(f"Fs RF = {fs_rf/1e6:.3f} MS/s, muestras = {n_total}")

    sdr = RtlSdr()
    sdr.sample_rate = fs_rf
    sdr.center_freq = fc_mhz * 1e6
    sdr.gain = gain
    sdr.rtl_agc = False

    chunks = []
    remaining = n_total

    try:
        while remaining > 0:
            n = min(block, remaining)
            chunks.append(sdr.read_samples(n))
            remaining -= n
            done = 100 * (1 - remaining / n_total)
            print(f"\rProgreso: {done:5.1f} %", end="")
    finally:
        sdr.close()

    print("\nLectura finalizada.")

    iq = np.concatenate(chunks).astype(np.complex64)
    np.savez_compressed(out, iq=iq, fs_rf=float(fs_rf), station_hz=float(fc_mhz * 1e6), gain_db=float(gain))

    print(f"Captura guardada en: {out}")


def process_to_fm_mpx(iq, fs_rf):
    n = np.arange(len(iq))
    iq_shifted = iq  # shift Hz = 0 para estación centrada

    lp_fm = firwin(161, BW_FM, pass_zero=True, fs=fs_rf)
    iq_filt = fir_filter_delay(iq_shifted, lp_fm)

    iq_limited = iq_filt / (np.abs(iq_filt) + 1e-6)

    # 2.048 MHz -> 228 kHz:
    # 2.048e6 * 57 / 512 = 228e3
    iq_bb = resample_poly(iq_limited, up=57, down=512)

    fm = np.angle(iq_bb[1:] * np.conj(iq_bb[:-1]))
    fm = fm - np.mean(fm)

    return fm


def extract_rds_19k(fm, fs=FS_MPX, route_name="piloto_19k_al_cubo"):
    """
    Extrae RDS desde el multiplex FM y lo lleva a 19 kHz.

    route_name:
      - "piloto_19k_al_cubo": genera la referencia de 57 kHz a partir
        del piloto estéreo de 19 kHz.
      - "oscilador_fijo_57k": usa un oscilador local fijo de 57 kHz.
    """
    bp_rds = firwin(
        NUMTAPS_SYNC,
        [RDS_BPF_LOW, RDS_BPF_HIGH],
        pass_zero=False,
        fs=fs,
        window=("kaiser", 8.0)
    )
    rds_band = fir_filter_delay(fm, bp_rds)

    if route_name == "piloto_19k_al_cubo":
        bp_pilot = firwin(
            NUMTAPS_SYNC,
            [18.7e3, 19.3e3],
            pass_zero=False,
            fs=fs,
            window=("kaiser", 8.0)
        )
        pilot = fir_filter_delay(fm, bp_pilot)
        pilot_a = hilbert(pilot)
        pilot_unit = pilot_a / (np.abs(pilot_a) + 1e-12)
        lo_57 = pilot_unit ** 3

        N = min(len(rds_band), len(lo_57))
        rds_band = rds_band[:N]
        lo_57 = lo_57[:N]

    elif route_name == "oscilador_fijo_57k":
        N = len(rds_band)
        t = np.arange(N) / fs
        lo_57 = np.exp(1j * 2 * np.pi * RDS_FC * t)

    else:
        raise ValueError("Ruta RDS no reconocida: " + str(route_name))

    rds_bb = rds_band * np.conj(lo_57)

    lp_rds = firwin(
        NUMTAPS_SYNC,
        RDS_LPF_HZ,
        pass_zero=True,
        fs=fs,
        window=("kaiser", 8.0)
    )
    rds_bb = fir_filter_delay(rds_bb, lp_rds)

    discard = int(0.10 * fs)
    if len(rds_bb) > discard:
        rds_bb = rds_bb[discard:]

    rds_bb = normalize_power(rds_bb)

    # 228 kHz -> 19 kHz
    rds_19k = resample_poly(rds_bb, up=1, down=12)
    rds_19k = normalize_power(rds_19k)

    rds_corr, foff = estimate_residual_freq_bpsk_fft(rds_19k, FS_RDS, search_hz=300)
    rds_corr = rds_corr[int(0.15 * FS_RDS):]

    return rds_band, rds_bb, rds_corr, foff


def analyze_file(infile, savefigs=False, figdir="figuras_rds_A"):
    t0_all = tic()

    data = np.load(infile)
    iq = data["iq"]
    fs_rf = float(data["fs_rf"])
    station_hz = float(data["station_hz"])
    gain_db = float(data["gain_db"]) if "gain_db" in data.files else np.nan

    print("============================================================")
    print("PARTE A - ANÁLISIS PEDAGÓGICO RDS/BPSK")
    print("============================================================")
    print(f"Archivo:      {infile}")
    print(f"Estación FM: {station_hz/1e6:.3f} MHz")
    print(f"Fs RF:       {fs_rf/1e6:.3f} MS/s")
    print(f"Ganancia:    {gain_db} dB")
    print(f"Muestras:    {len(iq)}")

    if abs(fs_rf - FS_RF_DEFAULT) > 1:
        print("Advertencia: el script está optimizado para FS_RF=2.048 MHz.")

    if savefigs:
        Path(figdir).mkdir(parents=True, exist_ok=True)

    t0 = tic()
    fm = process_to_fm_mpx(iq, fs_rf)
    toc(t0, "FM demod + MPX a 228 kHz")

    print(f"Duración MPX: {len(fm)/FS_MPX:.2f} s")

    t0 = tic()
    # Ruta principal usada para las figuras de espectro/tiempo.
    rds_band, rds_bb, rds_corr, foff = extract_rds_19k(fm, route_name="piloto_19k_al_cubo")
    toc(t0, "Extracción RDS 57 kHz -> 19 kHz (ruta piloto^3)")
    print(f"Offset residual RDS estimado, ruta piloto^3: {foff:.2f} Hz")

    # PSD MPX y zoom RDS
    f_psd, Pxx = welch(fm, fs=FS_MPX, nperseg=16384, noverlap=8192, return_onesided=True)

    fig, axs = plt.subplots(2, 1, figsize=(11, 8))
    mask = (f_psd >= 0) & (f_psd <= 75e3)
    axs[0].semilogy(f_psd[mask] / 1e3, Pxx[mask])
    axs[0].axvline(19, linestyle="--", label="Piloto 19 kHz")
    axs[0].axvline(38, linestyle=":", label="Subportadora estéreo 38 kHz")
    axs[0].axvline(57, linestyle="--", label="RDS 57 kHz")
    axs[0].set_title("Multiplex FM demodulado: espectro 0--75 kHz")
    axs[0].set_xlabel("Frecuencia [kHz]")
    axs[0].set_ylabel("PSD")
    axs[0].grid(True)
    axs[0].legend()

    mask_zoom = (f_psd >= 52e3) & (f_psd <= 62e3)
    axs[1].semilogy(f_psd[mask_zoom] / 1e3, Pxx[mask_zoom])
    axs[1].axvline(57, linestyle="--", label="Centro RDS 57 kHz")
    axs[1].axvspan(RDS_BPF_LOW / 1e3, RDS_BPF_HIGH / 1e3, alpha=0.15, label="Filtro BPF RDS")
    axs[1].set_title("Zoom de la subportadora RDS")
    axs[1].set_xlabel("Frecuencia [kHz]")
    axs[1].set_ylabel("PSD")
    axs[1].grid(True)
    axs[1].legend()

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    if savefigs:
        fig.savefig(Path(figdir) / "A1_espectro_mpx_y_rds.png", dpi=160)

    # Espectrograma de MPX
    f_spec, t_spec, Sxx = spectrogram(fm, fs=FS_MPX, nperseg=4096, noverlap=2048, mode="psd")
    spec_mask = (f_spec >= 0) & (f_spec <= 75e3)
    fig = plt.figure(figsize=(11, 5))
    plt.imshow(
        10 * np.log10(Sxx[spec_mask, :] + 1e-14),
        aspect="auto",
        origin="lower",
        extent=[t_spec[0], t_spec[-1], f_spec[spec_mask][0]/1e3, f_spec[spec_mask][-1]/1e3],
    )
    plt.axhline(57, linestyle="--")
    plt.title("Espectrograma del multiplex FM: presencia temporal de RDS")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [kHz]")
    plt.colorbar(label="PSD [dB]")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    if savefigs:
        fig.savefig(Path(figdir) / "A2_espectrograma_mpx_rds.png", dpi=160)

    # Etapas temporales de acondicionamiento RDS.
    # Esta figura NO pretende mostrar una BPSK ideal todavía. Su objetivo es
    # aclarar qué ocurre antes del correlador de bits:
    #   1) señal pasabanda filtrada alrededor de 57 kHz;
    #   2) bajada a banda base compleja I/Q;
    #   3) señal a 19 kHz antes de extraer bits.
    fig, axs = plt.subplots(
        3, 1, figsize=(12.5, 9),
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.2]}
    )

    # Panel 1: pasabanda alrededor de 57 kHz. Se muestra una ventana corta,
    # porque todavía contiene una oscilación rápida de periodo ~17.5 us.
    n1 = min(int(0.02 * FS_MPX), len(rds_band))
    t1_ms = np.arange(n1) / FS_MPX * 1e3
    y1 = np.real(rds_band[:n1])
    y1 = y1 / (np.percentile(np.abs(y1), 95) + 1e-12)
    axs[0].plot(t1_ms, y1, linewidth=1.0)
    axs[0].set_title("1) RDS pasabanda filtrado alrededor de 57 kHz: todavía NO son bits")
    axs[0].set_xlabel("Tiempo [ms] - ventana corta por la portadora de 57 kHz")
    axs[0].set_ylabel("Amplitud norm.")
    axs[0].grid(True)

    # Panel 2: banda base compleja I/Q. La portadora de 57 kHz ya se quitó;
    # por eso puede mostrarse una ventana más larga. I y Q conservan fase.
    n2 = min(int(0.020 * FS_MPX), len(rds_bb))
    t2_ms = np.arange(n2) / FS_MPX * 1e3
    i2 = np.real(rds_bb[:n2])
    q2 = np.imag(rds_bb[:n2])
    scale2 = np.percentile(np.abs(np.r_[i2, q2]), 95) + 1e-12
    axs[1].plot(t2_ms, i2 / scale2, label="I(t): componente en fase", linewidth=1.0)
    axs[1].plot(t2_ms, q2 / scale2, label="Q(t): componente en cuadratura", linewidth=1.0, alpha=0.85)
    axs[1].set_title("2) RDS bajado a banda base compleja I/Q: se conserva la información de fase")
    axs[1].set_xlabel("Tiempo [ms] - ventana más larga porque ya no está la portadora de 57 kHz")
    axs[1].set_ylabel("Amplitud norm.")
    axs[1].grid(True)
    axs[1].legend(loc="best")

    # Panel 3: señal a 19 kHz previa al correlador bifase. Se marcan
    # separaciones aproximadas de bit para conectar con la figura A4/A5.
    bit_period_ms = 1e3 / 1187.5
    n3 = min(int(0.020 * FS_RDS), len(rds_corr))
    t3_ms = np.arange(n3) / FS_RDS * 1e3
    y3 = np.real(rds_corr[:n3])
    y3 = y3 / (np.percentile(np.abs(y3), 95) + 1e-12)
    axs[2].plot(t3_ms, y3, linewidth=1.0)
    for tb in np.arange(0, t3_ms[-1] if len(t3_ms) else 0, bit_period_ms):
        axs[2].axvline(tb, linestyle=":", linewidth=0.7, alpha=0.6)
    axs[2].set_title("3) RDS a 19 kHz antes de decisión: aquí se aplicará el correlador bifase")
    axs[2].set_xlabel("Tiempo [ms] - líneas punteadas: intervalos aproximados de bit")
    axs[2].set_ylabel("I norm.")
    axs[2].grid(True)

    fig.suptitle("Etapas temporales de extracción RDS: pasabanda → I/Q banda base → señal para bits", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    if savefigs:
        fig.savefig(Path(figdir) / "A3_etapas_extraccion_rds.png", dpi=170)
        # Se mantiene el nombre antiguo por compatibilidad con la guía.
        fig.savefig(Path(figdir) / "A3_extraccion_rds_tiempo.png", dpi=170)

    # Validación RDS y selección de candidato para constelación.
    # A diferencia de versiones anteriores, aquí se prueban varias rutas
    # de bajada a banda base. Esto evita que el análisis quede amarrado a
    # una sola emisora o a una sola forma de recuperar la subportadora.
    all_groups = []
    candidate_results = []
    corr_candidates = build_corr_candidates()

    print(f"Modo búsqueda RDS Parte A: {SEARCH_MODE} (genérico si AUTO)")
    print(f"Candidatos de correlador por ruta: {len(corr_candidates)}")
    if SEARCH_MODE.upper().strip() == "AUTO":
        print("AUTO: no se priorizan parámetros de una emisora específica.")
    print(f"Rutas RDS: {RDS_ROUTES}")

    ps_name = "????????"
    ps_seen = [False, False, False, False]
    pi_dom = None

    for route_name in RDS_ROUTES:
        print("\n------------------------------------------------------------")
        print(f"Ruta RDS Parte A: {route_name}")
        print("------------------------------------------------------------")

        # Para no recalcular, reutilizamos la ruta piloto^3 ya calculada.
        if route_name == "piloto_19k_al_cubo":
            rds_corr_route = rds_corr
            foff_route = foff
        else:
            t0_route = tic()
            _, _, rds_corr_route, foff_route = extract_rds_19k(fm, route_name=route_name)
            toc(t0_route, f"Extracción RDS ruta {route_name}")

        print(f"Offset residual estimado en ruta {route_name}: {foff_route:.2f} Hz")

        for sps, off in corr_candidates:
            result = evaluate_candidate(rds_corr_route, sps_bit=sps, bit_offset=off)
            if result is None:
                continue

            result["route_name"] = route_name
            all_groups.extend(result["all_groups"])
            candidate_results.append(result)

            groups_tmp, pi_tmp = consolidate_groups(all_groups)
            ps_tmp, ps_seen_tmp = vote_ps(groups_tmp)

            print(
                f"Candidato {route_name} | sps={sps:.3f}, off={off:.2f}: "
                f"grupos mejor modo={result['n_groups']:3d}, "
                f"grupos todas variantes={result['n_groups_all_variants']:3d}, "
                f"PS acumulado={ps_tmp}"
            )

            ps_name = ps_tmp
            ps_seen = ps_seen_tmp
            pi_dom = pi_tmp

            if STOP_WHEN_PS_COMPLETE_PART_A and all(ps_seen):
                print("PS completo encontrado en Parte A. Se detiene la búsqueda.")
                break

        if STOP_WHEN_PS_COMPLETE_PART_A and all(ps_seen):
            break

    groups_cons, pi_dom = consolidate_groups(all_groups)
    ps_name, ps_seen = vote_ps(groups_cons)

    print("\n============================================================")
    print("VALIDACIÓN RDS")
    print("============================================================")
    print(f"Grupos brutos encontrados: {len(all_groups)}")
    print(f"Grupos consolidados:      {len(groups_cons)}")
    print("Nota: los grupos brutos se obtienen acumulando todas las variantes")
    print("      de inversión/diferencial de cada candidato, como en Parte B.")
    if pi_dom is not None:
        print(f"PI dominante:             0x{pi_dom:04X}")
    else:
        print("PI dominante:             no disponible")
    print(f"PS parcial/consolidado:   {ps_name}")
    print(f"Segmentos PS recibidos:   {ps_seen}")
    if all(ps_seen):
        print("Interpretación PS:        completo")
    elif any(ps_seen):
        print("Interpretación PS:        parcial pero consistente con RDS real")
    else:
        print("Interpretación PS:        sin segmentos suficientes")

    valid_rds = len(groups_cons) >= MIN_GROUPS_FOR_VALID_RDS

    if not valid_rds:
        print("\nNo se validó RDS.")
        print("Se mostrarán espectros y señales filtradas, pero NO se reportará constelación BPSK válida.")
        print("Esto evita confundir ruido filtrado con una constelación BPSK real.")

    if candidate_results and valid_rds:
        best = max(candidate_results, key=lambda r: (r["n_groups_all_variants"], r["n_groups"]))
        metrics = best["metrics"]
        bits = best["bits"]
        groups_best = best["groups"]

        # Figura didáctica principal: cómo se extraen bits desde la señal temporal.
        bit_start_for_plot = int(groups_best[0]["pos"]) if groups_best else 0
        bit_start_for_plot = max(0, bit_start_for_plot - 4)
        plot_bit_extraction_time(
            symbol_signal=best["symbol_signal"],
            metrics=metrics,
            raw_phase_bits=best["raw_phase_bits"],
            decoded_bits=bits,
            sps_bit=best["sps_bit"],
            bit_offset=best["bit_offset"],
            bit_start=bit_start_for_plot,
            n_bits=24,
            threshold=best["threshold"],
            groups_best=groups_best,
            savefigs=savefigs,
            figdir=figdir
        )

        # ------------------------------------------------------------
        # Visualización de BPSK validada por grupos RDS
        # ------------------------------------------------------------
        raw_phase_bits = best["raw_phase_bits"]
        threshold = best["threshold"]
        I_all = np.real(metrics)

        if groups_best:
            first_group_pos = int(groups_best[0]["pos"])
        else:
            first_group_pos = 0

        # Selección de puntos asociados a grupos RDS válidos para no confundir
        # ruido filtrado con símbolos BPSK reales.
        idx_valid = gather_valid_metric_indices(metrics, groups_cons, group_len=104, max_groups=8)
        m_rot, conf_mask = rotate_bpsk_metrics_for_display(metrics, raw_phase_bits, idx_valid)

        # Ventana temporal compacta alrededor del primer grupo válido.
        w0 = max(0, first_group_pos - 10)
        w1 = min(len(metrics), w0 + 80)
        idx = np.arange(w0, w1)
        I_seg = np.real(m_rot[w0:w1])
        phase_seg = np.where(raw_phase_bits[w0:w1] > 0, 1.0, -1.0)
        bits_seg = bits[w0:w1]

        # Constellation with only confident points from validated groups.
        idx_conf = idx_valid[conf_mask[idx_valid]] if len(idx_valid) else np.array([], dtype=int)
        if len(idx_conf) < 20:
            idx_conf = idx_valid
        m_conf = m_rot[idx_conf]
        lab_conf = raw_phase_bits[idx_conf]

        # Figura clara: se separan métrica, fase y bit para no mezclar escalas.
        # Todas comparten el mismo eje horizontal: índice de bit k.
        fig = plt.figure(figsize=(14, 8.2))
        gs = fig.add_gridspec(
            3, 2,
            width_ratios=[1.35, 1.0],
            height_ratios=[1.0, 1.0, 1.0],
            hspace=0.35,
            wspace=0.28
        )

        ax_m = fig.add_subplot(gs[0, 0])
        ax_f = fig.add_subplot(gs[1, 0], sharex=ax_m)
        ax_b = fig.add_subplot(gs[2, 0], sharex=ax_m)
        ax_c = fig.add_subplot(gs[:, 1])

        # 1) Métrica: variable continua. El umbral de decisión es y=0.
        ax_m.plot(idx, I_seg, marker="o", markersize=3, linewidth=1.0, label="métrica $m_k$")
        ax_m.axhline(0, linestyle="--", linewidth=1.0, label="umbral de decisión")
        ax_m.set_ylabel("Métrica")
        ax_m.set_title("1) Métrica de decisión por bit")
        ax_m.grid(True)
        ax_m.legend(loc="best")

        # 2) Fase: variable discreta. No comparte escala con la métrica.
        ax_f.step(idx, phase_seg, where="mid", linewidth=1.4)
        ax_f.set_ylim(-1.25, 1.25)
        ax_f.set_yticks([-1, 1])
        ax_f.set_yticklabels(["fase π", "fase 0"])
        ax_f.set_ylabel("Fase BPSK")
        ax_f.set_title("2) Fase estimada a partir del signo de la métrica")
        ax_f.grid(True)

        # 3) Bit: variable binaria final después de inversión/diferencial.
        ax_b.step(idx, bits_seg, where="mid", linewidth=1.4)
        ax_b.set_ylim(-0.2, 1.2)
        ax_b.set_yticks([0, 1])
        ax_b.set_ylabel("Bit")
        ax_b.set_xlabel("Índice de bit $k$")
        ax_b.set_title("3) Bit RDS estimado")
        ax_b.grid(True)

        for ax in [ax_m, ax_f, ax_b]:
            for xg in [first_group_pos, first_group_pos + 26, first_group_pos + 52, first_group_pos + 78, first_group_pos + 104]:
                if w0 <= xg <= w1:
                    ax.axvline(xg, linestyle=":", linewidth=0.9)

        pos = lab_conf > 0
        neg = ~pos
        ax_c.scatter(np.real(m_conf[pos]), np.imag(m_conf[pos]), s=22, alpha=0.70, marker="o", label="fase 0 medida")
        ax_c.scatter(np.real(m_conf[neg]), np.imag(m_conf[neg]), s=22, alpha=0.70, marker="x", label="fase π medida")
        ax_c.scatter([1, -1], [0, 0], s=180, marker="+", linewidths=2.2, label="puntos ideales BPSK")
        ax_c.annotate("ideal 0", (1, 0), xytext=(1.08, 0.15))
        ax_c.annotate("ideal π", (-1, 0), xytext=(-1.55, 0.15))
        ax_c.axhline(0, linewidth=0.8)
        ax_c.axvline(0, linewidth=0.8)
        ax_c.grid(True)
        ax_c.axis("equal")
        ax_c.set_xlim(-1.9, 1.9)
        ax_c.set_ylim(-1.0, 1.0)
        ax_c.set_xlabel("Componente en fase I")
        ax_c.set_ylabel("Componente en cuadratura Q")
        ax_c.set_title(f"Constelación BPSK validada por RDS\n{len(groups_cons)} grupos, PI=0x{pi_dom:04X}")
        ax_c.legend(loc="best")

        fig.suptitle("BPSK RDS: métrica, fase, bit y constelación validada", fontsize=13)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        if savefigs:
            fig.savefig(Path(figdir) / "A5_bpsk_tiempo_y_constelacion_clara.png", dpi=170)

        # Primer grupo RDS válido y límites de bloques
        if groups_best:
            g0 = groups_best[0]
            start = int(g0["pos"])
            seq = bits[start:start+104]
            fig = plt.figure(figsize=(12, 3.8))
            plt.step(np.arange(len(seq)), seq, where="post")
            for x in [26, 52, 78]:
                plt.axvline(x, linestyle="--")
            plt.ylim(-0.2, 1.2)
            plt.grid(True)
            plt.xlabel("Bit dentro del grupo RDS")
            plt.ylabel("Bit")
            gt = (g0["B"] >> 12) & 0xF
            gv = "A" if ((g0["B"] >> 11) & 1) == 0 else "B"
            plt.title(f"Trama RDS validada: grupo {gt}{gv}, 4 bloques de 26 bits")
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])
            if savefigs:
                fig.savefig(Path(figdir) / "A6_trama_rds_104_bits.png", dpi=160)
    else:
        # Figura explícita de advertencia para estaciones sin RDS validado.
        fig = plt.figure(figsize=(9, 4))
        plt.axis("off")
        txt = (
            "No se generó constelación BPSK válida.\n\n"
            "Criterio usado:\n"
            f"  grupos RDS consolidados >= {MIN_GROUPS_FOR_VALID_RDS}\n\n"
            "Motivo: una estación sin RDS o con muy baja recepción puede producir nubes\n"
            "similares al filtrar ruido alrededor de 57 kHz. La constelación solo se\n"
            "acepta cuando hay grupos RDS validados por síndrome."
        )
        plt.text(0.03, 0.90, txt, va="top", family="monospace", fontsize=11.5)
        plt.title("Resultado de validación RDS")
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        if savefigs:
            fig.savefig(Path(figdir) / "A5_sin_constelacion_validada.png", dpi=160)

    print(f"\nTiempo total Parte A: {time.perf_counter() - t0_all:.2f} s")

    plt.show()



# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

if __name__ == "__main__":
    iq_file = resolve_iq_filename()

    print("============================================================")
    print("CONFIGURACIÓN PARTE A")
    print("============================================================")
    print(f"STATION_MHZ: {STATION_MHZ:.3f} MHz")
    print(f"IQ file:     {iq_file}")
    print(f"CAPTURE_NEW: {CAPTURE_NEW}")

    if CAPTURE_NEW or not Path(iq_file).exists():
        print("\n============================================================")
        print("CAPTURA RTL-SDR")
        print("============================================================")
        capture_rtl_sdr(
            fc_mhz=STATION_MHZ,
            seconds=CAPTURE_SECONDS,
            gain=GAIN_DB,
            out=iq_file,
            fs_rf=FS_RF_DEFAULT
        )
    else:
        check_capture_matches_station(iq_file, STATION_MHZ)

    print("\n============================================================")
    print("ANÁLISIS PARTE A")
    print("============================================================")
    analyze_file(
        infile=iq_file,
        savefigs=SAVEFIGS,
        figdir=FIGDIR
    )
