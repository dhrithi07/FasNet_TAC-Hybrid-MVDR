import pyroomacoustics as pra
import soundfile as sf
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

def room_mix_symmetric_fixed(files, micsetup='circular', plot=True, rt60=0.1):
    target_fs = 16000 # Standard for FaSNet-TAC

    if not os.path.exists(files[0]) or not os.path.exists(files[1]):
        print(f"Error: Files not found.")
        return

    # --- FIX: Load with forced resampling ---
    # librosa.load ensures the audio is the correct speed before entering the room
    audio0, fs0 = librosa.load(files[0], sr=target_fs)
    audio1, fs1 = librosa.load(files[1], sr=target_fs)

    # Match lengths
    min_len = min(len(audio0), len(audio1))
    audio0, audio1 = audio0[:min_len], audio1[:min_len]

    room_dim = [5, 4, 2.5]
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Create room with the target_fs
    room = pra.ShoeBox(
        room_dim, fs=target_fs, materials=pra.Material(e_absorption), max_order=max_order
    )

    # Symmetric Positions
    room.add_source([1.5, 1.5, 1.5], signal=audio1)
    room.add_source([3.5, 2.5, 1.5], signal=audio0)

    # Microphones
    center = [2.5, 2.0, 1.2]
    if micsetup == 'circular':
        mic_locs = pra.circular_2D_array(center=center[:2], M=4, phi0=0, radius=0.1)
        mic_locs = np.concatenate((mic_locs, np.ones((1, 4)) * center[2]), axis=0)
        room.add_microphone_array(mic_locs)

    print("Simulating at 16kHz...")
    room.simulate()

    # --- FIX: Use soundfile for writing to avoid bit-depth issues ---
    output_filename = "mix_4ch_tac.wav"
    signal = room.mic_array.signals.T

    # Peak normalization
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.9

    # Write as float32 or int16
    sf.write(output_filename, signal, target_fs)
    print(f"Saved mix to: {output_filename} at {target_fs}Hz")

    # Also save the clean references at the same sample rate for evaluation
    sf.write("ref1_16k.wav", audio0, target_fs)
    sf.write("ref2_16k.wav", audio1, target_fs)

if __name__ == "__main__":
    files = ('ref1.wav', 'ref2.wav')
    room_mix_symmetric_fixed(files)