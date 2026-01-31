# Multichannel Blind Source Separation (FaSNet-TAC + Hybrid MVDR)

This project implements a high-fidelity pipeline for **Blind Source Separation (BSS)** of audio signals. It leverages the symmetric properties of a multichannel microphone array to separate two overlapping speakers in a simulated acoustic environment.

## Project Components

The pipeline consists of three primary modules:

1. **`Multichannel_Symmetric_Audio_Mixer.py`**: Simulates a "perfectly symmetric" acoustic room using `pyroomacoustics`. Two sources are placed equidistant from a 4-channel circular microphone array to test the model's ability to handle spatial ambiguity.
2. **`FasNet_TAC_Hybrid_MVDR.py`**: The core separation engine. It uses a pretrained **FaSNet-TAC** neural network to generate initial estimates, which are then refined using an **MVDR (Minimum Variance Distortionless Response)** beamformer and a Time-Frequency masking polish.
3. **`evaluation_metrics.py`**: Quantifies performance using the `mir_eval` library, calculating **SDR** (Signal-to-Distortion Ratio), **SIR** (Signal-to-Interference Ratio), and **SAR** (Signal-to-Artifacts Ratio).

---

## Getting Started

Installation
To set up the environment, run the following commands. Note that asteroid is the primary library used to load the pre-trained FaSNet-TAC weights.

Bash
# Install Asteroid for FaSNet-TAC model access and source separation utilities
pip install asteroid

# Install other core dependencies for signal processing and evaluation
pip install torch pyroomacoustics soundfile mir_eval librosa scipy

### 2. Prepare References

Place your two clean reference audio files in the root directory:

* `ref1.wav` (Speaker 1)
* `ref2.wav` (Speaker 2)

### 3. Execution Flow

#### Step A: Generate the Mixture

Run the mixer to create a 4-channel spatial mixture at a synchronized 16kHz sample rate.

```bash
python Multichannel_Symmetric_Audio_Mixer.py

```

*This produces `mix_4ch_tac.wav` and resampled references `ref1_16k.wav`/`ref2_16k.wav`.*

#### Step B: Separate the Sources

Process the mixture through the Hybrid MVDR pipeline.

```bash
python FasNet_TAC_Hybrid_MVDR.py

```

*This produces `fasnet_final_1.wav` and `fasnet_final_2.wav`.*

#### Step C: Evaluate Performance

Calculate the objective metrics to verify separation quality.

```bash
python evaluation_metrics.py

```

---

## Methodology

### Symmetric Room Configuration

The simulation uses a  meter room. By placing speakers at exactly m from the array center, we evaluate the model's robustness against symmetric phase arrival.

### Hybrid Post-Processing

To achieve a high **SAR (Signal-to-Artifacts Ratio)**, the pipeline applies:

* **MVDR Beamforming**: Reduces spatial interference while preserving the target speaker's magnitude.
* **Time-Frequency Polishing**: A soft-masking approach with a **0.55 floor** to prevent the "musical noise" or robotic artifacts common in deep learning separation.

---

##  Expected Results

Typical performance on symmetric mixtures:
| Metric | Expected Value | Description |
| :--- | :--- | :--- |
| **SIR** | ~16.0 dB | High separation; minimal "bleed" from the other speaker. |
| **SAR** | ~15.0 dB | High audio fidelity; minimal robotic distortion. |
| **SDR** | ~8.5 dB | Overall signal quality. |

---



