# 📻 FM RDS BPSK SDR Lab

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![RTL-SDR](https://img.shields.io/badge/RTL--SDR-compatible-orange)
![DSP](https://img.shields.io/badge/DSP-FM%20%7C%20RDS%20%7C%20BPSK-green)
![RDS](https://img.shields.io/badge/RDS-57%20kHz-purple)
![Status](https://img.shields.io/badge/status-educational%20lab-yellow)
![License](https://img.shields.io/badge/license-MIT-lightgrey)


## Overview

Educational SDR laboratory for extracting and decoding **Radio Data System (RDS)** information from a commercial FM broadcast station. The RDS subcarrier is isolated from the FM multiplex, translated to baseband, synchronized, interpreted as a **BPSK** signal, and decoded into valid RDS groups.

This repository follows a technical and educational style: real IQ capture, FM demodulation, spectral analysis, FIR filtering, carrier recovery, BPSK visualization, bit extraction, RDS block validation, Program Identification (PI), Program Service name (PS), and RadioText (RT) reconstruction.

> The `figuras_rds_A/` folder name is intentionally preserved because the analysis script writes figures there by default. The included images are placeholders and should be replaced by the real plots generated after running Part A.

## 📂 Contents

-  **src** → Python code for SDR capture, demodulation, and spectrum analysis    


## 🧰 Hardware Setup

The measurements were performed using a **low-cost RTL-SDR USB dongle** connected to a standard FM broadcast antenna.

### Device

- **SDR Receiver:** RTL-SDR (RTL2832U compatible)
- **Frequency Band:** FM broadcast band
- **Interface:** USB
- **Antenna:** Wideband FM antenna
- Local FM station with RDS transmission.

If an IQ capture is already available in `.npz` format, the RTL-SDR is not required.

The RTL-SDR is used exclusively for **IQ data acquisition**, while all signal processing is performed offline in Python.

<p align="center">
<img src="https://github.com/user-attachments/assets/d929ca5d-2eb0-4496-8d00-b7ccac389ebc" alt="SDR-RTL" width="400">
</p>


## 🚀 Main Scripts

| Script | Purpose |
|---|---|
| `src/part_a_rds_analysis.py` | Captures or loads IQ data, demodulates FM, analyzes the FM multiplex, extracts the RDS component, validates BPSK behavior, and generates didactic figures. |
| `src/part_b_rds_decoding.py` | Processes the IQ capture, searches for valid RDS groups, consolidates the dominant PI, reconstructs PS by segment voting, and extracts RadioText when enough segments are available. |

---

## 🎯 Laboratory Goals

- Capture or load IQ samples from an FM broadcast station.
- Demodulate FM to obtain the composite multiplex signal.
- Identify the main spectral components of the FM multiplex:
  - mono audio band,
  - 19 kHz stereo pilot,
  - stereo difference region,
  - 57 kHz RDS subcarrier.
- Isolate the RDS band using FIR filtering.
- Translate the RDS subcarrier to baseband.
- Recover the BPSK-like symbol stream.
- Extract a binary RDS bitstream.
- Validate RDS blocks using syndrome checks.
- Decode and consolidate:
  - PI: Program Identification,
  - PS: Program Service name,
  - RT: RadioText.

---

## 📡 RDS in the FM Multiplex

In FM broadcasting, RDS is transmitted around the **57 kHz** subcarrier inside the composite multiplex signal. This frequency is the third harmonic of the 19 kHz stereo pilot:

```math
57\,\text{kHz} = 3 \cdot 19\,\text{kHz}
```

The RDS symbol rate is:

```math
R_s = 1187.5\,\text{symbols/s}
```

In this implementation, the extracted RDS signal is resampled to:

```math
F_s = 19\,\text{kHz}
```

which gives:

```math
\frac{19000}{1187.5} = 16
```

samples per RDS bit/symbol interval.

---

## 🔄 Processing Pipeline


<p align="center">
<img src="https://github.com/user-attachments/assets/d17cb44d-b7ff-46bd-ac04-9cd472b718e4" alt="pipeline" width="600">
</p>

## 🧠 Key DSP Blocks

### 🔹 1. FM Demodulation

The FM multiplex is obtained from the phase difference between consecutive IQ samples:

```math
x_{FM}[n] = \angle\left(x[n] \cdot x^*[n-1]\right)
```

where `x[n]` is the complex IQ signal.

---

### 🔹 2. RDS Band Extraction

The RDS component is isolated using a bandpass filter around the 57 kHz subcarrier:

```math
54\,\text{kHz} \leq f \leq 60\,\text{kHz}
```

---

### 🔹 3. Baseband Translation

Two routes are implemented:

| Route | Description |
|---|---|
| `piloto_19k_al_cubo` | Uses the 19 kHz stereo pilot raised to the third power to generate a 57 kHz reference. |
| `oscilador_fijo_57k` | Uses a fixed 57 kHz complex oscillator. |

Using both routes improves robustness when different stations exhibit different RDS recovery behavior.

---

### 🔹 4. BPSK Carrier Recovery

A BPSK Costas loop is used to stabilize the recovered baseband phase. A PCA-based rotation is then applied to align the BPSK symbol cloud with the in-phase axis.

---

### 🔹 5. Biphase Bit Metric

The RDS bit decision metric compares the first and second halves of each bit interval:

```math
m_k =
\frac{1}{N_1}\sum_{n \in \text{first half}} x[n]
-
\frac{1}{N_2}\sum_{n \in \text{second half}} x[n]
```

The sign of this metric is used to estimate the corresponding bit state.

---

### 🔹 6. RDS Group Validation

Each RDS block has:

```text
16 information bits + 10 check bits = 26 bits
```

A complete RDS group contains four blocks:

```text
A | B | C/C' | D
```

Therefore, one full RDS group contains:

```math
4 \cdot 26 = 104\,\text{bits}
```

The decoder searches for valid A, B, C/C', and D blocks, filters by the dominant PI, removes duplicates, and consolidates the final information.

---

## 🖼️ Main Figures

The most relevant figures are reserved in `figuras_rds_A/`. Replace the placeholder images with the real figures generated by the script.

### 📊 FM Multiplex Spectrum and RDS Zoom

<p align="center">
<img src="https://github.com/user-attachments/assets/52885f3c-bbdd-46f3-8001-adc849a0646c" alt="RDS zoom" width="600">
</p>

###  FM Multiplex Spectrogram


<p align="center">
<img src="https://github.com/user-attachments/assets/9312ddc0-788a-43d1-80da-114d14b9794a" alt="espectro" width="600">
</p>


### 🧩 RDS Extraction Stages

![RDS extraction stages](figuras_rds_A/A3_etapas_extraccion_rds.png)


<p align="center">
<img src="https://github.com/user-attachments/assets/e54c2d56-cbb1-4b4e-8ef2-809ece1f6817" alt="extraction" width="600">
</p>

### 🔢 RDS Bitstream Extraction

<p align="center">
<img src="https://github.com/user-attachments/assets/1a76582f-2a48-4dd4-93af-097ec4e05ad0" alt="extraction2" width="600">
</p>

### 🟣 BPSK Time Signal and Validated Constellation

<p align="center">
<img src="https://github.com/user-attachments/assets/d194a9f7-1615-4cb5-a17c-efda64b234e2" alt="constelation" width="600">
</p>


### 🧾 Validated 104-bit RDS Group

<p align="center">
<img src="https://github.com/user-attachments/assets/c128151a-69d4-49a1-b16a-d8fb4e41cf0b" alt="constelation" width="600">
</p>


## 📦 Python Dependencies

Install the main dependencies:

```bash
pip install numpy scipy matplotlib
```

For direct RTL-SDR capture:

```bash
pip install pyrtlsdr
```

On Linux, RTL-SDR system libraries may also be required:

```bash
sudo apt install rtl-sdr librtlsdr-dev
```

## 🧪 Running Part A: Analysis and Figure Generation

Part A performs IQ capture or IQ loading, FM demodulation, multiplex analysis, RDS extraction, BPSK visualization, and figure generation.

```bash
python src/part_a_rds_analysis.py
```

Before running, review the configuration section at the beginning of the script:

```python
STATION_MHZ = 98.9
CAPTURE_SECONDS = 65.0
GAIN_DB = 30.0
CAPTURE_NEW = False
SAVEFIGS = True
```

Relevant parameters:

| Parameter | Description |
|---|---|
| `STATION_MHZ` | Selected FM station frequency. |
| `CAPTURE_SECONDS` | IQ capture duration. |
| `GAIN_DB` | RTL-SDR gain. |
| `CAPTURE_NEW` | If `True`, captures from RTL-SDR. If `False`, loads an existing capture. |
| `SAVEFIGS` | Saves the generated figures. |
| `AUTO_IQ_FILENAME` | Automatically names captures based on station frequency. |
| `STRICT_STATION_MATCH` | Verifies that the IQ capture matches the configured station. |

---

## 🧬 Running Part B: RDS Decoding

Part B processes the same IQ capture and attempts to recover valid RDS groups.

```bash
python src/part_b_rds_decoding.py
```

This stage reports:

- raw detected RDS groups,
- dominant PI,
- consolidated groups,
- PS segments,
- PS by voting,
- RadioText segments,
- reconstructed RadioText when enough segments are available.

---

## 🔍 Search Modes

The scripts include automatic synchronization search options so the lab is not tied to a single radio station.

```python
SEARCH_MODE = "AUTO"
```

For a previously characterized station, a preset mode can be used:

```python
SEARCH_MODE = "PRESET_CAROLINA"
```

For general laboratory use, `AUTO` is recommended.

---

## 💾 IQ Capture Naming

When `AUTO_IQ_FILENAME = True`, the IQ capture name is generated from the selected frequency.

Example:

```python
STATION_MHZ = 98.9
```

produces:

```text
fm_rds_iq_98_9MHz.npz
```

This avoids accidentally analyzing an old capture from a different station.

---

## ✅ Expected Output

A successful decoding run may produce a console summary similar to:

```text
============================================================
GLOBAL RESULT
============================================================

Raw detected groups: XX

Detected PI count:
  0xXXXX: XX

Dominant PI: 0xXXXX
Consolidated groups: XX

PS by voting:
  segment 0: 'XX' votes=X
  segment 1: 'XX' votes=X
  segment 2: 'XX' votes=X
  segment 3: 'XX' votes=X

Consolidated PS: 'XXXXXXXX'
PS segments: [True, True, True, True]

RadioText:
  group 2A segment 0: '....'
  group 2A segment 1: '....'

Consolidated RT: 'Recovered text from the FM station'
```



## 🧑‍🏫 Educational Use

This project can be used in courses or workshops related to:

- analog communications,
- digital communications,
- software-defined radio,
- FM demodulation,
- stereo FM multiplexing,
- FIR filtering,
- spectral analysis,
- BPSK modulation,
- carrier recovery,
- symbol synchronization,
- digital frame decoding,
- error detection and block validation.



## 🧰 Suggested Future Improvements

- Refactor the processing chain into reusable Python modules.
- Add command-line arguments with `argparse`.
- Export decoded PI, PS, and RT results to JSON or CSV.
- Add a Jupyter notebook for teaching demonstrations.
- Include a LaTeX laboratory guide.
- Compare multiple FM stations in a single report.
- Add unit tests for the RDS syndrome and block validation functions.
- Add an optional graphical interface for selecting frequency and capture settings.


## 📚 References

- GNU Radio – Official Documentation: [GNU Radio](https://www.gnuradio.org/)

- Frequency Modulation (FM) – Wikipedia: [Wikipedia Frequency Modulation](https://en.wikipedia.org/wiki/Frequency_modulation)

- FM broadcasting – Wikipedia: [FM broadcasting](https://en.wikipedia.org/wiki/FM_broadcasting)


## ⚠️ Disclaimer

This project is intended **for educational and experimental purposes only**.

It is provided to demonstrate signal processing concepts related to FM broadcast reception and spectrum analysis.  
The author does not encourage or endorse any unauthorized or improper use of radio equipment.

Users are responsible for ensuring compliance with **local laws and regulations** regarding radio reception and spectrum usage.

## 🤝 Support projects
 Support me on Patreon [https://www.patreon.com/c/CrissCCL](https://www.patreon.com/c/CrissCCL)

## 📜 License
MIT License  


