import torch
import functools

# --- CRITICAL FIX FOR PYTORCH 2.6 ---
# We must disable the strict security check for this specific model
# because it was trained in 2020 using older numpy formats.
# This patch forces weights_only=False globally.
print("Applying PyTorch 2.6+ security patch...")
torch.load = functools.partial(torch.load, weights_only=False)

# --- IMPORTS (Must come AFTER the patch) ---
import soundfile as sf
from asteroid.models import BaseModel
import numpy as np
import os
import time


def run_fasnet(input_file):
    print(f"--- Running FaSNet-TAC (with TTA) on {input_file} ---")

    # 1. Load Pre-trained FaSNet-TAC Model
    model_id = "popcornell/FasNetTAC-paper"
    print(f"Loading model '{model_id}'...")

    try:
        # The patch at the top of the file handles the 'weights_only' error automatically now
        model = BaseModel.from_pretrained(model_id)
        model.eval()
    except Exception as e:
        print(f"\nCRITICAL ERROR: Could not download/load model.")
        print(f"Details: {e}")
        return

    # 2. Load Multi-Channel Audio
    # Soundfile reads as (Samples, Channels) -> e.g. (Data, 4)
    data, fs = sf.read(input_file)
    print(f"Loaded audio with shape: {data.shape} and sample rate: {fs}")

    # 3. Prepare Data for Model
    # FaSNet expects shape: (Batch, Microphones, Time)

    # Transpose: (Samples, 4) -> (4, Samples)
    data = data.T

    # Convert to Tensor
    waveform = torch.from_numpy(data).float()

    # SAFETY NORMALIZATION (Input)
    # Scale input to 0.9 to prevent model overload
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.9
        print(f"Input normalized. Peak was: {max_val:.4f}")

    # Add Batch Dimension -> (1, 4, Samples)
    input_tensor = waveform.unsqueeze(0)

    # 4. Run Separation (UPDATED FOR TTA)
    print("Separating sources (using TTA / Phase Flipping)...")
    start_time = time.time()

    with torch.no_grad():
        # --- CHANGE START ---
        # Pass 1: Standard Inference
        print("   > Pass 1: Standard...")
        est_normal = model(input_tensor)

        # Pass 2: Inverted Phase Inference
        # We flip the audio upside down (-input), run it, then flip the result back.
        # This forces the model to analyze the wave from a different perspective.
        print("   > Pass 2: Inverted...")
        est_flipped = model(-input_tensor)
        est_inverted = -est_flipped

        # Average the two results to cancel out errors
        print("   > Averaging results...")
        estimated_sources = (est_normal + est_inverted) / 2.0
        # --- CHANGE END ---

    print(f"--- Processing Time: {time.time() - start_time:.4f} seconds ---")

    # 5. Save Outputs
    # Remove batch dimension -> (Sources, Time)
    output_tensor = estimated_sources.squeeze(0)

    for i in range(output_tensor.shape[0]):
        # Get single source
        out_audio = output_tensor[i].numpy()

        # SAFETY NORMALIZATION (Output)
        # Prevents "Blue Block" clipping
        max_out = np.max(np.abs(out_audio))
        if max_out > 0:
            out_audio = out_audio / max_out * 0.9

        fname = f"fasnet_output_{i + 1}.wav"
        sf.write(fname, out_audio, 16000)
        print(f"Saved: {fname}")


if __name__ == "__main__":
    # Use the relative path since we are running from the project folder
    filename = "mix_4ch_tac.wav"

    # Simple check
    if os.path.exists(filename):
        run_fasnet(filename)
    else:
        print(f"Error: '{filename}' not found.")
        print("Please run your mixing script (Multichannel_Symmetric_Audio_Mixer.py) first.")

#apply MVDR
import numpy as np
import soundfile as sf
from scipy import signal
import os


def apply_mvdr_hybrid(mix_file, est_files):
    print("--- Applying MVDR-Hybrid (Spatial + Spectral) ---")

    mix, fs = sf.read(mix_file)
    if mix.ndim == 1:
        print("Error: Mix must be 4-channel.")
        return

    est1, _ = sf.read(est_files[0])
    est2, _ = sf.read(est_files[1])

    min_len = min(len(mix), len(est1), len(est2))
    mix = mix[:min_len].T
    ests = np.stack([est1[:min_len], est2[:min_len]], axis=0)

    n_fft, hop = 1024, 512
    f, t, Zxx_mix = signal.stft(mix, fs, nperseg=n_fft, noverlap=hop)
    num_mics, num_freq, num_frames = Zxx_mix.shape

    # Pre-calculate Spectrograms for Masking
    _, _, Zxx_s1 = signal.stft(ests[0], fs, nperseg=n_fft, noverlap=hop)
    _, _, Zxx_s2 = signal.stft(ests[1], fs, nperseg=n_fft, noverlap=hop)

    # THE KEY: Create a Wiener Mask to help the MVDR
    mag1, mag2 = np.abs(Zxx_s1) ** 2, np.abs(Zxx_s2) ** 2
    mask1 = mag1 / (mag1 + mag2 + 1e-9)
    mask2 = mag2 / (mag1 + mag2 + 1e-9)
    masks = [mask1, mask2]

    outputs = []
    for s_idx in range(2):
        print(f"   > Processing Source {s_idx + 1}...")
        target_mask = masks[s_idx]
        processed_spec = np.zeros((num_freq, num_frames), dtype=complex)

        for f_idx in range(num_freq):
            X = Zxx_mix[:, f_idx, :]  # (4, Frames)

            # Use the Mask to estimate the Noise Covariance (Phi_NN)
            # This tells the MVDR exactly what to 'cancel'
            noise_mask = 1 - target_mask[f_idx, :]
            X_noise = X * noise_mask
            Rnn = np.dot(X_noise, X_noise.conj().T) / (np.sum(noise_mask) + 1e-9)
            Rnn += np.eye(num_mics) * 1e-5  # Regularization

            # Estimate Steering Vector from the Target-Masked signal
            X_target = X * target_mask[f_idx, :]
            Rss = np.dot(X_target, X_target.conj().T) / (np.sum(target_mask[f_idx, :]) + 1e-9)
            eigvals, eigvecs = np.linalg.eigh(Rss)
            d = eigvecs[:, -1]  # Principal component is the steering vector

            try:
                Rinv = np.linalg.inv(Rnn)
                w = np.dot(Rinv, d) / (np.dot(d.conj().T, np.dot(Rinv, d)) + 1e-9)

                # Apply MVDR beamformer
                spatial_out = np.dot(w.conj().T, X)

                # POST-FILTERING: Apply a light mask to ensure they don't sound the same
                processed_spec[f_idx, :] = spatial_out * (target_mask[f_idx, :] * 0.8 + 0.2)

            except np.linalg.LinAlgError:
                processed_spec[f_idx, :] = Zxx_s1[f_idx, :] if s_idx == 0 else Zxx_s2[f_idx, :]

        outputs.append(processed_spec)

    for i, spec in enumerate(outputs):
        _, audio = signal.istft(spec, fs, nperseg=n_fft, noverlap=hop)
        audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.9
        sf.write(f"fasnet_mvdr_hybrid_{i + 1}.wav", audio, fs)


if __name__ == "__main__":
    apply_mvdr_hybrid("mix_4ch_tac.wav", ["fasnet_output_1.wav", "fasnet_output_2.wav"])

#Apply MVDR polish
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter
import os


def apply_sar_polish(mix_file, mvdr_files):
    # --- CALIBRATED FOR SAR 33 / SIR 9 ---
    alpha = 1.3  # Slightly increased to keep separation sharp
    sigma = 1.0  # Smoothing to remove digital 'jitter'
    floor = 0.40  # Increased to 40%
    # -------------------------------------

    print(f"--- Polishing MVDR for SAR 30+ | Floor: {floor} ---")

    def load_mono(f):
        d, fs = sf.read(f)
        if d.ndim > 1: d = d[:, 0]
        return d, fs

    mix, fs = load_mono(mix_file)
    m1, _ = load_mono(mvdr_files[0])
    m2, _ = load_mono(mvdr_files[1])

    min_len = min(len(mix), len(m1), len(m2))
    mix, m1, m2 = mix[:min_len], m1[:min_len], m2[:min_len]

    # STFT
    nperseg, noverlap = 1024, 768
    f, t, Zxx_mix = signal.stft(mix, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_m1 = signal.stft(m1, fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_m2 = signal.stft(m2, fs, nperseg=nperseg, noverlap=noverlap)

    # 1. Soft Ratio Mask
    eps = np.finfo(float).eps
    mag_1 = np.abs(Zxx_m1) ** alpha
    mag_2 = np.abs(Zxx_m2) ** alpha

    mask_1 = mag_1 / (mag_1 + mag_2 + eps)
    mask_2 = mag_2 / (mag_1 + mag_2 + eps)

    # 2. Gaussian Smoothing
    mask_1 = gaussian_filter(mask_1, sigma=sigma)
    mask_2 = gaussian_filter(mask_2, sigma=sigma)

    # 3. Apply Quality Floor
    mask_1 = np.maximum(mask_1, floor)
    mask_2 = np.maximum(mask_2, floor)

    # 4. Apply to MVDR signals
    # This keeps the spatial benefits of MVDR but softens the artifacts
    Zxx_out1 = Zxx_m1 * mask_1
    Zxx_out2 = Zxx_m2 * mask_2

    # 5. Inverse STFT
    _, out1 = signal.istft(Zxx_out1, fs, nperseg=nperseg, noverlap=noverlap)
    _, out2 = signal.istft(Zxx_out2, fs, nperseg=nperseg, noverlap=noverlap)

    def save_norm(fname, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0: audio = audio / max_val * 0.9
        sf.write(fname, audio, fs)

    save_norm("fasnet_final_1.wav", out1)
    save_norm("fasnet_final_2.wav", out2)
    print("✅ Polish Complete.")


if __name__ == "__main__":
    mvdr_files = ["fasnet_mvdr_hybrid_1.wav", "fasnet_mvdr_hybrid_2.wav"]
    if os.path.exists(mvdr_files[0]):
        apply_sar_polish("mix_4ch_tac.wav", mvdr_files)
    else:
        print("Error: MVDR files (fasnet_hybrid) not found. Run MVDR script first.")