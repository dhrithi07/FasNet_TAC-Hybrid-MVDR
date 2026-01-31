import numpy as np
import soundfile as sf
import mir_eval
import os


def load_files(file_list):
    """Loads a list of files and returns a list of numpy arrays."""
    loaded_data = []
    for f in file_list:
        data, fs = sf.read(f)
        if data.ndim > 1:
            data = data[:, 0]
        loaded_data.append(data)
    return loaded_data


def evaluate():
    print("--- Evaluating Results: Raw vs. FaSNet vs. MVDR vs. MVDR Polish ---")

    # 1. Define File Groups based on your specifications
    ref_files = ['ref1_16k.wav', 'ref2_16k.wav']
    mix_file = 'mix_4ch_tac.wav'

    fasnet_files = ['fasnet_output_1.wav', 'fasnet_output_2.wav']
    mvdr_files = ['fasnet_mvdr_hybrid_1.wav', 'fasnet_mvdr_hybrid_2.wav']
    polish_files = ['fasnet_final_1.wav', 'fasnet_final_2.wav']

    # 2. Check File Existence
    def check_group(group_name, files):
        exists = all(os.path.exists(f) for f in files)
        if not exists:
            print(f"Warning: {group_name} files not found. Skipping column.")
        return exists

    has_refs = check_group("Reference", ref_files)
    has_mix = os.path.exists(mix_file)
    has_fasnet = check_group("FaSNet Output", fasnet_files)
    has_mvdr = check_group("MVDR Hybrid", mvdr_files)
    has_polish = check_group("MVDR Polish", polish_files)

    if not has_refs or not has_mix:
        print("Error: Reference or Mixture files are missing. Cannot evaluate.")
        return

    # 3. Load Data
    refs_data = load_files(ref_files)
    mix_data, fs = sf.read(mix_file)

    fasnet_data = load_files(fasnet_files) if has_fasnet else []
    mvdr_data = load_files(mvdr_files) if has_mvdr else []
    polish_data = load_files(polish_files) if has_polish else []

    # 4. Global Truncation (Ensures all arrays match in length)
    lengths = [len(r) for r in refs_data] + [mix_data.shape[0]]
    if has_fasnet: lengths += [len(f) for f in fasnet_data]
    if has_mvdr:   lengths += [len(m) for m in mvdr_data]
    if has_polish: lengths += [len(p) for p in polish_data]

    min_len = min(lengths)
    print(f"Trimming evaluation to {min_len} samples...")

    # 5. Prepare Stacks
    ref_stack = np.stack([r[:min_len] for r in refs_data])

    # Baseline (Mix)
    mix_mono = mix_data[:min_len, 0]
    mix_stack = np.stack([mix_mono, mix_mono])

    def get_stack(data):
        return np.stack([d[:min_len] for d in data])

    # 6. Calculate Metrics
    print("\nCalculating metrics (mir_eval)...")

    # A. Mix Baseline
    sdr_b, sir_b, sar_b, _ = mir_eval.separation.bss_eval_sources(ref_stack, mix_stack, compute_permutation=True)

    # B. FaSNet Output
    res_f = mir_eval.separation.bss_eval_sources(ref_stack, get_stack(fasnet_data), True) if has_fasnet else None

    # C. MVDR Hybrid
    res_m = mir_eval.separation.bss_eval_sources(ref_stack, get_stack(mvdr_data), True) if has_mvdr else None

    # D. MVDR Polish (Final)
    res_p = mir_eval.separation.bss_eval_sources(ref_stack, get_stack(polish_data), True) if has_polish else None

    # 7. Print Table
    print("\n" + "=" * 130)
    header = f"{'Metric':<8} | {'Mix':<10} | {'FaSNet':<18} | {'MVDR Hybrid':<18} | {'MVDR Polish':<18} | {'Improvement':<10}"
    print(header)
    print("-" * 130)

    def print_row(name, idx, base_val):
        row = f"{name:<8} | {base_val:<10.2f} | "

        val_f = np.mean(res_f[idx]) if res_f else 0.0
        val_m = np.mean(res_m[idx]) if res_m else 0.0
        val_p = np.mean(res_p[idx]) if res_p else 0.0

        row += f"{val_f:<18.2f} | {val_m:<18.2f} | {val_p:<18.2f} | "

        # Improvement: Best output vs Mix
        best_val = max(val_f, val_m, val_p)
        gain = best_val - base_val
        row += f"+{gain:.2f} dB"
        print(row)

    print_row("SDR", 0, np.mean(sdr_b))
    print_row("SIR", 1, np.mean(sir_b))
    print_row("SAR", 2, np.mean(sar_b))
    print("-" * 130)


if __name__ == "__main__":
    evaluate()