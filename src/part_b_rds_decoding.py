import time
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, resample_poly, welch, hilbert


# ============================================================
# PARTE B - DECODIFICACIÓN RDS
# ============================================================
# Objetivo:
#   En una sola ejecución sobre una captura suficiente, recuperar RDS
#   acumulando grupos válidos desde varios sincronizadores/candidatos.
#
# Luego:
#   - filtra por PI dominante,
#   - elimina duplicados,
#   - reconstruye PS por votación de segmentos,
#   - reconstruye RT por segmentos.
#
# Requisito:
#   fm_rds_iq_capture.npz generado por Parte A.
# ============================================================


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Debe coincidir con la frecuencia usada en Parte A.
STATION_MHZ =98.9 # <-----coloque aqui la estacion de radio elegida 

IQ_FILENAME = "fm_rds_iq_capture.npz"
AUTO_IQ_FILENAME = True
STRICT_STATION_MATCH = True

FS_MUX = 228e3
BW_FM = 120e3
SHIFT_HZ = 0

PILOT_FC = 19e3
RDS_FC = 57e3

RDS_SYMBOL_RATE = 1187.5
FS_RDS = 19e3
SPS = int(FS_RDS / RDS_SYMBOL_RATE)  # 16 muestras/símbolo

NUMTAPS_SYNC = 2001
RDS_BPF_LOW = 54.0e3
RDS_BPF_HIGH = 60.0e3
RDS_LPF_HZ = 3.8e3

# Búsqueda M&M local, acotada pero robusta.
MM_INTERP = 32
MM_GAINS = [0.005, 0.010, 0.020, 0.050]
MM_INPUT_OFFSETS = list(range(0, SPS, 2))  # 0,2,...14

# Costas
COSTAS_ALPHA = 0.030
COSTAS_BETA = 0.00025

# Correlador bifase por bit.
SPS_SEARCH_CORR = np.array([15.98, 16.00, 16.02])
N_OFFSETS_CORR = 12

# ------------------------------------------------------------
# Modo de búsqueda de RDS
# ------------------------------------------------------------
# AUTO:
#   explora una grilla genérica de sps/offset.
#   No usa candidatos preajustados para una emisora específica.
SEARCH_MODE = "AUTO"

AUTO_SPS_VALUES = [15.98, 16.00, 16.02]
AUTO_OFFSET_DIVISIONS = 12

def build_corr_params_for_route(route_name):
    """
    Construye candidatos genéricos para el correlador bifase.

    Esta versión no usa presets por emisora. Para cada ruta RDS se prueba
    la misma grilla de sps/offset:
      - sps cerca de 16 muestras/bit
      - offset dentro de un intervalo de bit completo
    """
    grid = []
    for sps in AUTO_SPS_VALUES:
        for off in np.linspace(0, 16.0, AUTO_OFFSET_DIVISIONS, endpoint=False):
            grid.append((float(sps), float(off)))

    return grid


MM_PARAMS_BY_ROUTE = {
    # v13: M&M omitido. En las corridas exitosas, los grupos útiles vinieron del correlador.
}

# Para evitar falsos positivos, por defecto no corregir 1 bit.
# Si quieres explorar más, puedes cambiar a True, pero revisa PI dominante.
ALLOW_ONE_BIT_CORRECTION = False

# Si quieres filtrar por un PI conocido luego de haberlo identificado:
# EXPECTED_PI = 0xCB1A
EXPECTED_PI = None

# No se usa para validar RDS. Solo se imprime como ayuda didáctica
# si falta un segmento PS y el patrón coincide fuertemente.
PS_AUTOCOMPLETE_HINTS = []

# Para acelerar pruebas, puedes limitar segundos de captura.
# None usa todo el archivo.
PROCESS_SECONDS = None

# Para comportamiento tipo radio: detener apenas se complete el PS.
# Si quieres acumular más RadioText, cambia a False.
STOP_WHEN_PS_COMPLETE = True

# Si True, después de completar PS continúa hasta alcanzar este número de segmentos RT.
CONTINUE_FOR_RT_AFTER_PS = True
RT_TARGET_SEGMENTS = 12



def station_filename_from_mhz(station_mhz):
    txt = f"{station_mhz:.1f}".replace(".", "_")
    return f"fm_rds_iq_{txt}MHz.npz"


def resolve_iq_filename():
    if AUTO_IQ_FILENAME:
        return station_filename_from_mhz(STATION_MHZ)
    return IQ_FILENAME


def check_capture_matches_station(infile, station_mhz):
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
            "Ejecute nuevamente la Parte A con CAPTURE_NEW=True para generar\n"
            "la captura de esta emisora, o ajuste STATION_MHZ/IQ_FILENAME.\n"
        )
        if STRICT_STATION_MATCH:
            raise RuntimeError(msg)
        print(msg)
        return False

    return True


# ============================================================
# UTILIDADES
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
    return x / (np.sqrt(np.mean(np.abs(x)**2)) + 1e-12)


def estimate_residual_freq_bpsk_fft(x, fs, search_hz=300):
    x = x - np.mean(x)
    z = x**2

    N = len(z)
    nfft = 2**int(np.ceil(np.log2(N)))
    window = np.hanning(N)

    Z = np.fft.fftshift(np.fft.fft(z * window, n=nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/fs))

    mask = (f >= -2 * search_hz) & (f <= 2 * search_hz)
    f_search = f[mask]
    Z_search = Z[mask]

    f2_est = f_search[np.argmax(np.abs(Z_search))]
    f_est = f2_est / 2.0

    n = np.arange(len(x))
    y = x * np.exp(-1j * 2 * np.pi * f_est * n / fs)

    return y, f_est


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

    return y, angle, center


# ============================================================
# RDS: SÍNDROMES Y GRUPOS
# ============================================================

def bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def rds_syndrome_lfsr(block_bits):
    poly = 0x5B9
    reg = 0

    for b in block_bits:
        reg = (reg << 1) | int(b)
        if reg & (1 << 10):
            reg ^= poly

    return reg & 0x3FF


RDS_OFFSETS_LFSR = {
    0x0FC: "A",
    0x198: "B",
    0x168: "C",
    0x350: "C'",
    0x1B4: "D",
}

# Matriz H del artículo de Friedt.
H_ARTICLE = np.array([
    [1,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,1,0,0],
    [0,1,0,1,1,0,1,1,1,0],
    [0,0,1,0,1,1,0,1,1,1],
    [1,0,1,0,0,0,0,1,1,1],
    [1,1,1,0,0,1,1,1,1,1],
    [1,1,0,0,0,1,0,0,1,1],
    [1,1,0,1,0,1,0,1,0,1],
    [1,1,0,1,1,1,0,1,1,0],
    [0,1,1,0,1,1,1,0,1,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,0,1,1,1,0,0],
    [0,1,1,1,1,0,1,1,1,0],
    [0,0,1,1,1,1,0,1,1,1],
    [1,0,1,0,1,0,0,1,1,1],
    [1,1,1,0,0,0,1,1,1,1],
    [1,1,0,0,0,1,1,0,1,1],
], dtype=np.uint8)

ARTICLE_SYNDROMES = {
    tuple([1,1,1,1,0,1,1,0,0,0]): "A",
    tuple([1,1,1,1,0,1,0,1,0,0]): "B",
    tuple([1,0,0,1,0,1,1,1,0,0]): "C",
    tuple([1,1,1,1,0,0,1,1,0,0]): "C'",
    tuple([1,0,0,1,0,1,1,0,0,0]): "D",
    tuple([0,1,0,1,0,1,1,0,0,0]): "D",
}


def rds_syndrome_article(block_bits):
    b = np.array(block_bits, dtype=np.uint8)
    return tuple((b @ H_ARTICLE) % 2)


def classify_block(block_bits, allow_one_bit=False):
    block_bits = np.array(block_bits, dtype=np.uint8)

    s_lfsr = rds_syndrome_lfsr(block_bits)
    if s_lfsr in RDS_OFFSETS_LFSR:
        return RDS_OFFSETS_LFSR[s_lfsr], bits_to_int(block_bits[:16]), block_bits, 0, "lfsr"

    s_h = rds_syndrome_article(block_bits)
    if s_h in ARTICLE_SYNDROMES:
        return ARTICLE_SYNDROMES[s_h], bits_to_int(block_bits[:16]), block_bits, 0, "H"

    if allow_one_bit:
        for i in range(26):
            b2 = block_bits.copy()
            b2[i] ^= 1

            s_lfsr = rds_syndrome_lfsr(b2)
            if s_lfsr in RDS_OFFSETS_LFSR:
                return RDS_OFFSETS_LFSR[s_lfsr], bits_to_int(b2[:16]), b2, 1, "lfsr1"

            s_h = rds_syndrome_article(b2)
            if s_h in ARTICLE_SYNDROMES:
                return ARTICLE_SYNDROMES[s_h], bits_to_int(b2[:16]), b2, 1, "H1"

    return None, None, None, None, None


def find_rds_groups(bitstream, allow_one_bit=False, max_groups=300):
    """
    Busca grupos usando A, B y D como condición fuerte.
    C/C' se reporta si valida, pero no se exige para sincronizar.
    """
    groups = []
    N = len(bitstream)

    for i in range(0, N - 104 + 1):
        bA = bitstream[i:i+26]
        bB = bitstream[i+26:i+52]
        bC = bitstream[i+52:i+78]
        bD = bitstream[i+78:i+104]

        tA, dA, _, eA, mA = classify_block(bA, allow_one_bit=allow_one_bit)
        if tA != "A":
            continue

        tB, dB, _, eB, mB = classify_block(bB, allow_one_bit=allow_one_bit)
        if tB != "B":
            continue

        tD, dD, _, eD, mD = classify_block(bD, allow_one_bit=allow_one_bit)
        if tD != "D":
            continue

        tC, dC, _, eC, mC = classify_block(bC, allow_one_bit=allow_one_bit)

        errors = (eA or 0) + (eB or 0) + (eD or 0) + (eC or 0)
        methods = [mA, mB, mC, mD]

        groups.append({
            "pos": i,
            "A": dA,
            "B": dB,
            "C": dC if tC in ("C", "C'") else None,
            "D": dD,
            "C_type": tC if tC in ("C", "C'") else None,
            "errors": errors,
            "methods": methods
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


# ============================================================
# DECODIFICACIÓN PS Y RT
# ============================================================

def printable_byte(v):
    return chr(v) if 32 <= v <= 126 else "?"


def decode_ps_name(groups):
    ps_chars = ["?"] * 8
    seen = [False] * 4
    pi_code = None
    decoded_segments = []

    for g in groups:
        A = g["A"]
        B = g["B"]
        D = g["D"]

        pi_code = A
        group_type = (B >> 12) & 0xF
        version = (B >> 11) & 0x1

        if group_type == 0:
            segment_addr = B & 0x03

            c1 = (D >> 8) & 0xFF
            c2 = D & 0xFF

            ch1 = printable_byte(c1)
            ch2 = printable_byte(c2)

            ps_chars[2 * segment_addr] = ch1
            ps_chars[2 * segment_addr + 1] = ch2
            seen[segment_addr] = True

            decoded_segments.append({
                "segment": segment_addr,
                "chars": ch1 + ch2,
                "version": "A" if version == 0 else "B"
            })

    return "".join(ps_chars), seen, pi_code, decoded_segments


def decode_radiotext(groups):
    rt_chars = ["?"] * 64
    seen = [False] * 16
    segments = []

    for g in groups:
        B = g["B"]
        C = g["C"]
        D = g["D"]

        group_type = (B >> 12) & 0xF
        version = (B >> 11) & 0x1

        if group_type != 2:
            continue

        addr = B & 0x0F

        if version == 0:
            base = 4 * addr
            chars = ["?", "?", "?", "?"]

            if C is not None:
                chars[0] = printable_byte((C >> 8) & 0xFF)
                chars[1] = printable_byte(C & 0xFF)

            chars[2] = printable_byte((D >> 8) & 0xFF)
            chars[3] = printable_byte(D & 0xFF)

            for i in range(4):
                if base + i < len(rt_chars):
                    rt_chars[base + i] = chars[i]

            seen[addr] = True
            segments.append({"type": "2A", "addr": addr, "chars": "".join(chars)})

        else:
            base = 2 * addr
            ch1 = printable_byte((D >> 8) & 0xFF)
            ch2 = printable_byte(D & 0xFF)

            if base < len(rt_chars):
                rt_chars[base] = ch1
            if base + 1 < len(rt_chars):
                rt_chars[base + 1] = ch2

            seen[addr] = True
            segments.append({"type": "2B", "addr": addr, "chars": "".join(chars)})

    rt_text = "".join(rt_chars).rstrip("?")
    return rt_text, seen, segments


# ============================================================
# RECONSTRUCCIÓN POR VOTACIÓN
# ============================================================

def dominant_pi(groups):
    counts = {}
    for g in groups:
        pi = g["A"]
        counts[pi] = counts.get(pi, 0) + 1
    if not counts:
        return None, {}
    pi = max(counts, key=counts.get)
    return pi, counts


def consolidate_groups(groups, expected_pi=None):
    """
    Filtra por PI dominante o esperado y elimina duplicados.
    """
    if not groups:
        return [], None, {}

    pi_dom, counts = dominant_pi(groups)
    pi_use = expected_pi if expected_pi is not None else pi_dom

    unique = {}

    for g in groups:
        if g["A"] != pi_use:
            continue

        key = (g["A"], g["B"], g["C"], g["D"], g.get("C_type", None))

        if key not in unique or g.get("errors", 0) < unique[key].get("errors", 999):
            unique[key] = g

    out = list(unique.values())
    out.sort(key=lambda g: g.get("pos", 0))
    return out, pi_use, counts


def vote_ps(groups):
    """
    Vota por segmento PS. Si el mismo segmento aparece varias veces, se elige
    el par de caracteres más frecuente.
    """
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
    detail = []

    for seg in range(4):
        if votes[seg]:
            best_chars = max(votes[seg], key=votes[seg].get)
            ps_pairs[seg] = best_chars
            seen[seg] = True
            detail.append((seg, best_chars, votes[seg][best_chars]))

    return "".join(ps_pairs), seen, detail


def vote_rt(groups):
    """
    Reconstrucción de RadioText por votación carácter a carácter.
    No deja que caracteres '?' de segmentos parciales sobreescriban caracteres válidos.
    """
    char_votes = [dict() for _ in range(64)]
    seen = [False] * 16
    segments = []

    for g in groups:
        B = g["B"]
        C = g["C"]
        D = g["D"]

        group_type = (B >> 12) & 0xF
        version = (B >> 11) & 0x1

        if group_type != 2:
            continue

        addr = B & 0x0F

        if version == 0:
            base = 4 * addr
            chars = ["?", "?", "?", "?"]

            if C is not None:
                chars[0] = printable_byte((C >> 8) & 0xFF)
                chars[1] = printable_byte(C & 0xFF)

            chars[2] = printable_byte((D >> 8) & 0xFF)
            chars[3] = printable_byte(D & 0xFF)

            for i, ch in enumerate(chars):
                pos = base + i
                if pos < 64 and ch != "?":
                    char_votes[pos][ch] = char_votes[pos].get(ch, 0) + 1

            seen[addr] = True
            segments.append({"type": "2A", "addr": addr, "chars": "".join(chars)})

        else:
            base = 2 * addr
            chars = [
                printable_byte((D >> 8) & 0xFF),
                printable_byte(D & 0xFF)
            ]

            for i, ch in enumerate(chars):
                pos = base + i
                if pos < 64 and ch != "?":
                    char_votes[pos][ch] = char_votes[pos].get(ch, 0) + 1

            seen[addr] = True
            segments.append({"type": "2B", "addr": addr, "chars": "".join(chars)})

    rt_chars = []
    for votes in char_votes:
        if votes:
            rt_chars.append(max(votes, key=votes.get))
        else:
            rt_chars.append("?")

    rt_text = "".join(rt_chars).rstrip("?")
    return rt_text, seen, segments


# ============================================================
# SINCRONIZADORES / CANDIDATOS
# ============================================================

def mm_clock_recovery(samples, sps=SPS, interp=32, mm_gain=0.01, input_offset=0, mu0=0.01):
    samples = np.asarray(samples, dtype=np.complex64)

    if input_offset > 0:
        samples = samples[input_offset:]

    samples_interpolated = resample_poly(samples, interp, 1)

    out = np.zeros(len(samples) // sps + 20, dtype=np.complex64)
    out_rail = np.zeros(len(samples) // sps + 20, dtype=np.complex64)

    i_in = 0
    i_out = 2
    mu = mu0

    max_i_out = len(out)
    max_i_in = len(samples) - 2

    while i_out < max_i_out and i_in < max_i_in:
        interp_index = i_in * interp + int(mu * interp)

        if interp_index < 0 or interp_index >= len(samples_interpolated):
            break

        out[i_out] = samples_interpolated[interp_index]

        out_rail[i_out] = (
            (1.0 if np.real(out[i_out]) >= 0 else -1.0)
            + 1j * (1.0 if np.imag(out[i_out]) >= 0 else -1.0)
        )

        x = (out_rail[i_out] - out_rail[i_out - 2]) * np.conj(out[i_out - 1])
        y = (out[i_out] - out[i_out - 2]) * np.conj(out_rail[i_out - 1])
        mm_val = np.real(y - x)

        mu += sps + mm_gain * mm_val

        step = int(np.floor(mu))
        i_in += step
        mu -= step

        i_out += 1

    return out[2:i_out]


def biphase_correlator_bits_frac(x_real, sps_bit, bit_offset, mapping=0):
    """
    Correlador bifase por bit completo, versión vectorizada.

    Antes esto se calculaba con un bucle por bit y era la parte más lenta.
    Ahora interpola todas las muestras de integración de una sola vez.
    """
    x = np.asarray(x_real, dtype=float)
    n = np.arange(len(x), dtype=float)

    n_bits = int((len(x) - bit_offset - sps_bit) // sps_bit)
    if n_bits <= 10:
        return np.array([], dtype=np.uint8), 0.0, 0.0

    m = 8
    frac_first = (np.arange(m) + 0.5) / (2 * m)
    frac_second = 0.5 + (np.arange(m) + 0.5) / (2 * m)

    starts = bit_offset + np.arange(n_bits)[:, None] * sps_bit
    idx1 = starts + frac_first[None, :] * sps_bit
    idx2 = starts + frac_second[None, :] * sps_bit

    v1 = np.interp(idx1.ravel(), n, x).reshape(n_bits, m)
    v2 = np.interp(idx2.ravel(), n, x).reshape(n_bits, m)

    first_vals = np.mean(v1, axis=1)
    second_vals = np.mean(v2, axis=1)

    metrics = first_vals - second_vals

    bits = (metrics > 0).astype(np.uint8)
    if mapping:
        bits ^= 1

    transition_pct = 100.0 * np.mean(np.sign(first_vals) != np.sign(second_vals))
    reliability = np.mean(np.abs(metrics)) / (np.std(metrics) + 1e-12)

    return bits, transition_pct, reliability


def decode_bitstream_all_variants(bits0, metadata):
    """
    Prueba variantes baratas y devuelve todos los grupos válidos encontrados.
    """
    found = []

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

                groups = find_rds_groups(
                    b,
                    allow_one_bit=ALLOW_ONE_BIT_CORRECTION,
                    max_groups=300
                )

                for g in groups:
                    gg = dict(g)
                    gg["_method"] = metadata.get("method", "")
                    gg["_route"] = metadata.get("route", "")
                    gg["_param"] = metadata.get("param", "")
                    gg["_mode"] = mode
                    gg["_final_inv"] = final_inv
                    found.append(gg)

    return found


def collect_groups_mm(rds_19k, route_name):
    """
    M&M + Costas + slicer con pocos parámetros ya conocidos.
    """
    all_groups = []
    candidate_summaries = []

    params = MM_PARAMS_BY_ROUTE.get(route_name, [])

    for input_offset, mm_gain in params:
        mm_symbols = mm_clock_recovery(
            rds_19k,
            sps=SPS,
            interp=MM_INTERP,
            mm_gain=mm_gain,
            input_offset=input_offset,
            mu0=0.01
        )

        if len(mm_symbols) < 300:
            continue

        locked = costas_loop_bpsk(mm_symbols, alpha=COSTAS_ALPHA, beta=COSTAS_BETA)
        locked = locked[50:]

        if len(locked) < 300:
            continue

        rot, angle, center = pca_rotate_bpsk(locked)

        soft_i = np.real(rot)
        bits0 = (soft_i > np.median(soft_i)).astype(np.uint8)

        pos = soft_i[bits0 == 1]
        neg = soft_i[bits0 == 0]

        if len(pos) > 10 and len(neg) > 10:
            sep = abs(np.median(pos) - np.median(neg))
            spread = np.std(pos) + np.std(neg) + 1e-12
            quality = sep / spread
        else:
            quality = 0.0

        meta = {
            "method": "MM",
            "route": route_name,
            "param": f"off={input_offset},gain={mm_gain},Q={quality:.2f}"
        }

        groups = decode_bitstream_all_variants(bits0, meta)
        all_groups.extend(groups)

        ps_n = 0
        rt_n = 0
        if groups:
            groups_cons, _, _ = consolidate_groups(groups, expected_pi=EXPECTED_PI)
            _, ps_seen, _ = vote_ps(groups_cons)
            _, rt_seen, _ = vote_rt(groups_cons)
            ps_n = sum(ps_seen)
            rt_n = sum(rt_seen)

        candidate_summaries.append({
            "route": route_name,
            "offset": input_offset,
            "gain": mm_gain,
            "quality": quality,
            "groups": len(groups),
            "ps_n": ps_n,
            "rt_n": rt_n
        })

    candidate_summaries.sort(
        key=lambda c: (c["groups"], c["ps_n"], c["rt_n"], c["quality"]),
        reverse=True
    )

    print(f"\nTop M&M targeted con grupos para {route_name}:")
    for c in candidate_summaries[:10]:
        print(
            f"  grupos={c['groups']:3d} ps={c['ps_n']} rt={c['rt_n']} | "
            f"off={c['offset']:2d} gain={c['gain']:.3f} Q={c['quality']:.2f}"
        )

    return all_groups, candidate_summaries


def collect_groups_correlator(rds_19k, route_name):
    """
    Costas + PCA + correlador bifase por bit completo.
    Versión rápida: usa solo parámetros que ya dieron grupos válidos.
    """
    all_groups = []
    summaries = []

    locked = costas_loop_bpsk(rds_19k, alpha=COSTAS_ALPHA, beta=COSTAS_BETA)
    locked = locked[int(0.25 * FS_RDS):]

    rot, angle, center = pca_rotate_bpsk(locked)
    x_real = np.real(rot)

    params = build_corr_params_for_route(route_name)

    for sps_bit, off in params:
        for mapping in [0]:
            bits0, transition_pct, reliability = biphase_correlator_bits_frac(
                x_real,
                sps_bit=sps_bit,
                bit_offset=off,
                mapping=mapping
            )

            if len(bits0) < 104:
                continue

            meta = {
                "method": "CORR",
                "route": route_name,
                "param": f"sps={sps_bit:.3f},off={off:.2f},map={mapping},tr={transition_pct:.1f},rel={reliability:.2f}"
            }

            groups = decode_bitstream_all_variants(bits0, meta)
            all_groups.extend(groups)

            ps_n = 0
            rt_n = 0
            if groups:
                groups_cons, _, _ = consolidate_groups(groups, expected_pi=EXPECTED_PI)
                _, ps_seen, _ = vote_ps(groups_cons)
                _, rt_seen, _ = vote_rt(groups_cons)
                ps_n = sum(ps_seen)
                rt_n = sum(rt_seen)

            summaries.append({
                "route": route_name,
                "sps": sps_bit,
                "off": off,
                "mapping": mapping,
                "groups": len(groups),
                "ps_n": ps_n,
                "rt_n": rt_n,
                "transition_pct": transition_pct,
                "reliability": reliability
            })

    summaries.sort(
        key=lambda c: (c["groups"], c["ps_n"], c["rt_n"], c["transition_pct"], c["reliability"]),
        reverse=True
    )

    print(f"\nTop correlador targeted con grupos para {route_name}:")
    for c in summaries[:10]:
        print(
            f"  grupos={c['groups']:3d} ps={c['ps_n']} rt={c['rt_n']} | "
            f"sps={c['sps']:.3f} off={c['off']:.2f} map={c['mapping']} "
            f"tr={c['transition_pct']:.1f}% rel={c['reliability']:.2f}"
        )

    return all_groups, summaries



def suggest_ps_completion(ps_name, hints):
    """
    Sugiere un PS probable si el texto decodificado tiene '?'.
    No reemplaza la decodificación real; solo ayuda cuando falta un segmento.
    """
    if "?" not in ps_name:
        return None

    best = None
    best_score = -1

    for h in hints:
        h = h[:8].ljust(8)
        score = 0
        conflict = False

        for a, b in zip(ps_name, h):
            if a == "?":
                continue
            if a == b:
                score += 1
            else:
                conflict = True
                break

        if not conflict and score > best_score:
            best_score = score
            best = h

    if best is not None and best_score >= 4:
        return best.strip()

    return None



def clean_rt_for_display(rt_text):
    rt_text = re.sub(r"\s+", " ", rt_text).strip()
    return rt_text


def collect_groups_correlator_param(rds_19k, route_name, sps_bit, off, mapping=0):
    """
    Ejecuta un único candidato del correlador bifase y devuelve grupos válidos.
    """
    locked = costas_loop_bpsk(rds_19k, alpha=COSTAS_ALPHA, beta=COSTAS_BETA)
    locked = locked[int(0.25 * FS_RDS):]

    rot, angle, center = pca_rotate_bpsk(locked)
    x_real = np.real(rot)

    bits0, transition_pct, reliability = biphase_correlator_bits_frac(
        x_real,
        sps_bit=sps_bit,
        bit_offset=off,
        mapping=mapping
    )

    if len(bits0) < 104:
        return [], {
            "route": route_name,
            "sps": sps_bit,
            "off": off,
            "mapping": mapping,
            "groups": 0,
            "ps_n": 0,
            "rt_n": 0,
            "transition_pct": transition_pct,
            "reliability": reliability,
        }

    meta = {
        "method": "CORR",
        "route": route_name,
        "param": f"sps={sps_bit:.3f},off={off:.2f},map={mapping},tr={transition_pct:.1f},rel={reliability:.2f}"
    }

    groups = decode_bitstream_all_variants(bits0, meta)

    ps_n = 0
    rt_n = 0
    if groups:
        groups_cons, _, _ = consolidate_groups(groups, expected_pi=EXPECTED_PI)
        _, ps_seen, _ = vote_ps(groups_cons)
        _, rt_seen, _ = vote_rt(groups_cons)
        ps_n = sum(ps_seen)
        rt_n = sum(rt_seen)

    summary = {
        "route": route_name,
        "sps": sps_bit,
        "off": off,
        "mapping": mapping,
        "groups": len(groups),
        "ps_n": ps_n,
        "rt_n": rt_n,
        "transition_pct": transition_pct,
        "reliability": reliability,
    }

    return groups, summary


# ============================================================
# GRÁFICOS
# ============================================================

def plot_summary(groups_cons, pi_use, processing_time):
    ps_name, ps_seen, ps_detail = vote_ps(groups_cons)
    rt_text, rt_seen, rt_segments = vote_rt(groups_cons)

    pi_text = "No disponible" if pi_use is None else f"0x{pi_use:04X}"

    text = (
        "RDS - Decodificación AUTO genérica\n\n"
        f"Tiempo de procesamiento: {processing_time:.2f} s\n"
        f"PI usado: {pi_text}\n"
        f"Grupos consolidados: {len(groups_cons)}\n"
        f"PS consolidado: {ps_name}\n"
        f"PS segmentos: {ps_seen}\n"
        f"RT consolidado: {clean_rt_for_display(rt_text)}\n"
        f"RT segmentos: {rt_seen}\n"
    )

    plt.figure(figsize=(10, 5.5))
    plt.axis("off")
    plt.text(0.04, 0.92, text, fontsize=11.5, va="top", family="monospace")
    plt.title("Resumen RDS - AUTO genérico")
    plt.tight_layout()


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

t_all = tic()

print("============================================================")
print("PARTE B - DECODIFICACIÓN RDS CON CONTROL DE CAPTURA")
print("============================================================")

t0 = tic()
IQ_FILENAME_RESOLVED = resolve_iq_filename()
check_capture_matches_station(IQ_FILENAME_RESOLVED, STATION_MHZ)

data = np.load(IQ_FILENAME_RESOLVED)
iq = data["iq"]
FS_RF = float(data["fs_rf"])
STATION_HZ = float(data["station_hz"])
gain_db = float(data["gain_db"]) if "gain_db" in data.files else np.nan
toc(t0, "Carga NPZ")

if PROCESS_SECONDS is not None:
    iq = iq[:int(PROCESS_SECONDS * FS_RF)]

print(f"Archivo:      {IQ_FILENAME_RESOLVED}")
print(f"Estación FM: {STATION_HZ/1e6:.3f} MHz")
print(f"Fs RF:       {FS_RF/1e6:.3f} MS/s")
print(f"Ganancia:    {gain_db} dB")
print(f"Muestras:    {len(iq)}")
print("Modo Parte B: búsqueda AUTO genérica; se detiene según configuración PS/RT.")
print(f"STATION_MHZ configurado: {STATION_MHZ:.3f} MHz")

if abs(FS_RF - 2.048e6) > 1:
    raise ValueError("Este script está ajustado para capturas con FS_RF = 2.048 MHz.")


# ------------------------------------------------------------
# FM a 228 kHz
# ------------------------------------------------------------

print("\nProcesando FM...")
t0 = tic()

n = np.arange(len(iq))
iq_shifted = iq * np.exp(-1j * 2 * np.pi * SHIFT_HZ * n / FS_RF)

lp_fm = firwin(161, BW_FM, pass_zero=True, fs=FS_RF)
iq_filt = fir_filter_delay(iq_shifted, lp_fm)

iq_limited = iq_filt / (np.abs(iq_filt) + 1e-6)

iq_bb = resample_poly(iq_limited, up=57, down=512)
fs = FS_MUX

fm = np.angle(iq_bb[1:] * np.conj(iq_bb[:-1]))
fm = fm - np.mean(fm)

toc(t0, "FM demod + resample a 228 kHz")
print(f"Duración procesada: {len(iq_bb)/fs:.2f} s")


# PSD básico
f_psd, Pxx = welch(fm, fs=fs, nperseg=16384, noverlap=8192, return_onesided=True)
mask = (f_psd >= 0) & (f_psd <= 75e3)

plt.figure(figsize=(10, 4))
plt.semilogy(f_psd[mask]/1e3, Pxx[mask], color="black")
plt.axvline(19, color="orange", linestyle="--", label="Piloto 19 kHz")
plt.axvline(57, color="violet", linestyle="--", label="RDS 57 kHz")
plt.grid(True)
plt.xlabel("Frecuencia [kHz]")
plt.ylabel("PSD")
plt.title("Parte B v18 - Multiplex FM a 228 kHz")
plt.legend()
plt.tight_layout()


# ------------------------------------------------------------
# Extraer piloto y RDS band
# ------------------------------------------------------------

print("\nExtrayendo piloto y banda RDS...")
t0 = tic()

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
lo_57_pilot = pilot_unit**3

bp_rds = firwin(
    NUMTAPS_SYNC,
    [RDS_BPF_LOW, RDS_BPF_HIGH],
    pass_zero=False,
    fs=fs,
    window=("kaiser", 8.0)
)
rds_band = fir_filter_delay(fm, bp_rds)

N = min(len(rds_band), len(lo_57_pilot))
rds_band = rds_band[:N]
lo_57_pilot = lo_57_pilot[:N]

t = np.arange(N) / fs
lo_57_fixed = np.exp(1j * 2 * np.pi * RDS_FC * t)

toc(t0, "Extracción piloto/RDS")


# ------------------------------------------------------------
# Procesar rutas RDS de forma adaptativa
# ------------------------------------------------------------

all_groups = []
all_summaries = []

ps_complete = False
rt_good_enough = False

for route_name, lo_57 in [
    ("piloto_19k_al_cubo", lo_57_pilot),
    ("oscilador_fijo_57k", lo_57_fixed),
]:
    if ps_complete and STOP_WHEN_PS_COMPLETE and not CONTINUE_FOR_RT_AFTER_PS:
        print("\nPS ya está completo. Se omite la ruta restante para ahorrar tiempo.")
        break

    print("\n============================================================")
    print(f"RUTA RDS: {route_name}")
    print("============================================================")

    t0 = tic()

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

    rds_19k = resample_poly(rds_bb, up=1, down=12)
    rds_19k = normalize_power(rds_19k)

    rds_corr, foff = estimate_residual_freq_bpsk_fft(rds_19k, FS_RDS, search_hz=300)
    rds_corr = rds_corr[int(0.15 * FS_RDS):]

    toc(t0, f"Ruta {route_name}: RDS a 19 kHz + offset")
    print(f"Offset residual estimado: {foff:.2f} Hz")
    print(f"Duración RDS útil: {len(rds_corr)/FS_RDS:.2f} s")

    print(f"\nCandidatos correlador adaptativo para {route_name}:")

    for sps_bit, off in build_corr_params_for_route(route_name):
        t0 = tic()

        groups_param, summary = collect_groups_correlator_param(
            rds_corr,
            route_name,
            sps_bit=sps_bit,
            off=off,
            mapping=0
        )

        all_groups.extend(groups_param)
        all_summaries.append(summary)

        groups_tmp, pi_tmp, _ = consolidate_groups(all_groups, expected_pi=EXPECTED_PI)
        ps_tmp, ps_seen_tmp, _ = vote_ps(groups_tmp)
        rt_tmp, rt_seen_tmp, _ = vote_rt(groups_tmp)

        toc(t0, f"  candidato sps={sps_bit:.3f}, off={off:.2f}")

        print(
            f"    grupos={summary['groups']:3d} "
            f"ps={summary['ps_n']} rt={summary['rt_n']} "
            f"tr={summary['transition_pct']:.1f}% rel={summary['reliability']:.2f}"
        )
        print(f"    acumulado PS='{ps_tmp}' segmentos={ps_seen_tmp} RTseg={sum(rt_seen_tmp)}")

        ps_complete = all(ps_seen_tmp)
        rt_good_enough = sum(rt_seen_tmp) >= RT_TARGET_SEGMENTS

        if ps_complete:
            print("    PS completo detectado.")
            if STOP_WHEN_PS_COMPLETE and not CONTINUE_FOR_RT_AFTER_PS:
                break

        if ps_complete and CONTINUE_FOR_RT_AFTER_PS and rt_good_enough:
            print("    PS completo y RT suficiente. Deteniendo búsqueda.")
            break

    if ps_complete and STOP_WHEN_PS_COMPLETE and not CONTINUE_FOR_RT_AFTER_PS:
        break

    if ps_complete and CONTINUE_FOR_RT_AFTER_PS and rt_good_enough:
        break


# ------------------------------------------------------------
# Consolidación
# ------------------------------------------------------------

t0 = tic()

groups_cons, pi_use, pi_counts = consolidate_groups(all_groups, expected_pi=EXPECTED_PI)
ps_name, ps_seen, ps_detail = vote_ps(groups_cons)
rt_text, rt_seen, rt_segments = vote_rt(groups_cons)

toc(t0, "Consolidación")

processing_time = time.perf_counter() - t_all


# ============================================================
# RESULTADOS
# ============================================================

print("\n============================================================")
print("RESULTADO GLOBAL - AUTO")
print("============================================================")

print(f"Total grupos brutos encontrados: {len(all_groups)}")

print("\nConteo de PI detectados:")
for pi, cnt in sorted(pi_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
    print(f"  0x{pi:04X}: {cnt}")

if pi_use is None:
    print("\nNo se encontraron grupos RDS válidos.")
else:
    print(f"\nPI usado/dominante: 0x{pi_use:04X}")
    print(f"Grupos consolidados: {len(groups_cons)}")

    print("\nPS por votación:")
    for seg, chars, count in ps_detail:
        print(f"  segmento {seg}: '{chars}'  votos={count}")

    print(f"\nPS consolidado: '{ps_name}'")
    print(f"Segmentos PS: {ps_seen}")
    if all(ps_seen):
        print("PS completo recuperado en una sola pasada.")

    ps_suggested = suggest_ps_completion(ps_name, PS_AUTOCOMPLETE_HINTS)
    if ps_suggested is not None and ps_suggested != ps_name:
        print(f"PS probable por patrón conocido: '{ps_suggested}'")
        print("Nota: esto es una sugerencia, no reemplaza segmentos RDS no recibidos.")

    print("\nRadioText:")
    for s in rt_segments[:30]:
        print(f"  grupo {s['type']} segmento {s['addr']}: '{s['chars']}'")
    print(f"RT consolidado: '{rt_text}'")
    rt_clean = clean_rt_for_display(rt_text)
    if rt_clean != rt_text:
        print(f"RT visual limpio: '{rt_clean}'")
    print(f"Segmentos RT: {rt_seen}")

    print("\nGrupos consolidados:")
    for g in groups_cons[:40]:
        group_type = (g["B"] >> 12) & 0xF
        version = "A" if ((g["B"] >> 11) & 1) == 0 else "B"
        print(
            f"  PI=0x{g['A']:04X} | grupo={group_type}{version} | "
            f"B=0x{g['B']:04X} "
            f"C={g['C'] if g['C'] is not None else '----'} "
            f"D=0x{g['D']:04X} | "
            f"err={g['errors']} | "
            f"{g.get('_method','')} {g.get('_route','')} {g.get('_param','')}"
        )

print(f"\nTiempo total: {processing_time:.2f} s")

plot_summary(groups_cons, pi_use, processing_time)

plt.show()
