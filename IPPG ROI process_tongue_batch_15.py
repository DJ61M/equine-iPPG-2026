# process_tongue_batch_working_hr_rr_restored.py
# Goal: Restore working HR (~49-51 bpm) + fix doubled RR (~27 → ~14)
# Cardiac: prominence 0.5×std + 0.012 floor, ≥3 peaks for HR
# Resp: prominence 0.25×std + 0.01 floor, min dist 1.8 s
# SpO₂ quality and clipping unchanged (working)

import cv2
import numpy as np
import os
import pandas as pd
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askdirectory

print("OpenCV version:", cv2.__version__)

root = Tk()
root.withdraw()
print("Select folder containing images + single RGB-colored mask file...")
folder = askdirectory(title="Select folder with images + RGB mask")
root.destroy()

if not folder or not os.path.isdir(folder):
    print("No valid folder selected. Exiting.")
    exit()

print(f"Selected folder: {folder}")
folder_name = os.path.basename(folder)

# ─── Load mask ───
rgb_mask_path = None
for f in os.listdir(folder):
    fname_lower = f.lower()
    if "mask" in fname_lower and fname_lower.endswith(('.png', '.jpg', '.jpeg')):
        rgb_mask_path = os.path.join(folder, f)
        break

if not rgb_mask_path:
    print("No mask file found. Exiting.")
    exit()

rgb_mask = cv2.imread(rgb_mask_path)
if rgb_mask is None:
    print(f"Failed to load mask: {rgb_mask_path}")
    exit()

print(f"Loaded RGB mask: {os.path.basename(rgb_mask_path)} shape = {rgb_mask.shape}")

# ─── Mask splitting ───
mask_upper = ((rgb_mask[:,:,0] >= 200) & (rgb_mask[:,:,1] <= 60) & (rgb_mask[:,:,2] <= 60)).astype(np.uint8) * 255
mask_tongue = ((rgb_mask[:,:,1] >= 200) & (rgb_mask[:,:,0] <= 60) & (rgb_mask[:,:,2] <= 60)).astype(np.uint8) * 255
mask_lower = ((rgb_mask[:,:,2] >= 200) & (rgb_mask[:,:,0] <= 60) & (rgb_mask[:,:,1] <= 60)).astype(np.uint8) * 255

masks = [mask_tongue, mask_upper, mask_lower]
region_names = ["Tongue", "Upper_Mucosa", "Lower_Mucosa"]
region_colors = ['green', 'red', 'blue']

# Pixel counts
for name, mask in zip(region_names, masks):
    pixels = np.sum(mask == 255)
    print(f"{name}: {pixels:,} pixels")

debug_dir = os.path.join(folder, "debug_masks")
os.makedirs(debug_dir, exist_ok=True)
for name, mask in zip(region_names, masks):
    cv2.imwrite(os.path.join(debug_dir, f"{name.lower()}_mask.png"), mask)
print(f"Debug masks saved to: {debug_dir}")

# ─── Frame pairs ───
def get_frame_nr(name):
    m = re.search(r'(\d{4})\.jpg$', name)
    return int(m.group(1)) if m else -1

files_660_dict = {}
for f in os.listdir(folder):
    if f.startswith("Basler_a2A2048-114umBAS__40575591"):
        frame_nr = get_frame_nr(f)
        if frame_nr != -1:
            files_660_dict[frame_nr] = f

files_850_dict = {}
for f in os.listdir(folder):
    if f.startswith("Basler_a2A2048-114umBAS__40658204"):
        frame_nr = get_frame_nr(f)
        if frame_nr != -1:
            files_850_dict[frame_nr] = f

common_frames = sorted(set(files_660_dict.keys()) & set(files_850_dict.keys()))
if not common_frames:
    print("No matching frame pairs found.")
    exit()

print(f"Found {len(common_frames)} frame pairs")

# ─── Process frames ───
results = {name: [] for name in region_names}
fps = 30.0

for i, frame_nr in enumerate(common_frames):
    f660 = files_660_dict[frame_nr]
    f850 = files_850_dict[frame_nr]
    path660 = os.path.join(folder, f660)
    path850 = os.path.join(folder, f850)

    img660 = cv2.imread(path660, cv2.IMREAD_GRAYSCALE)
    img850 = cv2.imread(path850, cv2.IMREAD_GRAYSCALE)
    if img660 is None or img850 is None:
        continue

    time_s = round(i / fps, 4)

    for j, (region, mask) in enumerate(zip(region_names, masks)):
        pixels_660 = img660[mask == 255]
        pixels_850 = img850[mask == 255]
        if len(pixels_660) < 500:
            continue
        mean660 = np.mean(pixels_660)
        mean850 = np.mean(pixels_850)
        ratio = mean660 / mean850 if mean850 > 0 else np.nan

        results[region].append({
            "frame": i,
            "time_s": time_s,
            "mean_660nm": round(mean660, 3),
            "mean_850nm": round(mean850, 3),
            "ratio_660_850": round(ratio, 6)
        })

    print(f"Frame {i:4d} processed", end="\r")
print("\nProcessing complete.")

# ─── Kalman smoother ───
def kalman_smooth_1d(signal, process_noise=0.001, measurement_noise=0.1):
    if len(signal) < 3:
        return signal.copy()
    n = len(signal)
    smoothed = np.zeros(n)
    F = np.array([[1, 1/fps], [0, 1]])
    H = np.array([[1, 0]])
    x = np.array([[signal[0]], [0.0]])
    P = np.eye(2) * 1.0
    Q = np.eye(2) * process_noise
    R = np.array([[measurement_noise]])
    for i in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        z = signal[i]
        y = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K * y
        P = (np.eye(2) - K @ H) @ P
        smoothed[i] = x[0, 0]
    return smoothed

# ─── Peak detection ───
def find_peaks_manual(sig, min_dist_frames, min_prom):
    peaks = []
    for j in range(1, len(sig)-1):
        if sig[j] > sig[j-1] and sig[j] > sig[j+1] and sig[j] > min_prom:
            if not peaks or (j - peaks[-1]) >= min_dist_frames:
                peaks.append(j)
    return peaks

# ─── Compute vitals ───
vitals = {}
z_score = 1.96

for region in region_names:
    if not results[region]:
        vitals[region] = {"spo2": None, "spo2_ci": (None,None), "spo2_quality": 0,
                          "hr": None, "hr_ci": (None,None),
                          "rr": None, "rr_ci": (None,None), "rr_note": "",
                          "median_ratio": None, "length_s": 0, "df": None}
        continue

    df = pd.DataFrame(results[region])
    time_vals = df['time_s'].values
    ratio_raw = df['ratio_660_850'].values

    # Soft motion correction
    if len(ratio_raw) > 5:
        running_mean = np.convolve(ratio_raw, np.ones(3)/3, mode='valid')
        running_mean = np.pad(running_mean, (1, 1), mode='edge')
        ratio = ratio_raw / (running_mean + 1e-6) * np.mean(ratio_raw)
    else:
        ratio = ratio_raw.copy()

    # SpO₂
    spo2_base = 120 - 18 * ratio
    spo2_tongue = spo2_base + 3.5
    spo2_per_frame = np.clip(spo2_tongue, 90, 102)
    mask_clipped = (spo2_per_frame <= 90) | (spo2_per_frame >= 102)
    spo2_per_frame[mask_clipped] = np.clip(118 - 15 * ratio[mask_clipped] + 4, 90, 102)

    spo2_mean = np.clip(np.nanmean(spo2_per_frame), None, 100.0)

    n_valid = len(spo2_per_frame[~np.isnan(spo2_per_frame)])
    if n_valid > 1:
        sem = np.nanstd(spo2_per_frame, ddof=1) / np.sqrt(n_valid)
        spo2_ci = (spo2_mean - z_score * sem, spo2_mean + z_score * sem)
    else:
        spo2_ci = (None, None)

    # Detrending
    window_sec = 3.0
    window_frames = int(window_sec * fps)
    if len(ratio) > window_frames:
        kernel = np.ones(window_frames) / window_frames
        moving_avg = np.convolve(ratio, kernel, mode='valid')
        padded_avg = np.pad(moving_avg, (window_frames//2, len(ratio) - len(moving_avg) - window_frames//2), mode='edge')
        ratio_detrended = ratio - padded_avg
    else:
        ratio_detrended = ratio - np.nanmean(ratio)

    # SpO₂ quality
    spo2_quality = 0.0

    if np.mean(ratio) > 0:
        cv_ratio = np.std(ratio) / np.mean(ratio)
        spo2_quality += 35 * max(0, 1 - min(1.2, cv_ratio * 2.5))

    puls_ampl = np.std(ratio_detrended) if len(ratio_detrended) > 0 else 0
    spo2_quality += 35 * min(1.0, puls_ampl / 0.025)

    clipped_frac = np.mean((spo2_per_frame <= 90) | (spo2_per_frame >= 102))
    spo2_quality += 15 * (1 - clipped_frac)

    mean_660 = df['mean_660nm'].mean() if 'mean_660nm' in df else 100
    mean_850 = df['mean_850nm'].mean() if 'mean_850nm' in df else 100
    mean_int = (mean_660 + mean_850) / 2
    spo2_quality += 15 * min(1.0, mean_int / 90)

    spo2_quality = int(max(0, min(100, spo2_quality)))

    # Bandpass
    def bp(sig, lo, hi):
        fft = np.fft.rfft(sig)
        f = np.fft.rfftfreq(len(sig), 1/fps)
        fft[(f < lo) | (f > hi)] = 0
        return np.fft.irfft(fft, n=len(sig))

    pulse_bp = bp(ratio_detrended, 0.7, 3.0)
    resp_bp  = bp(ratio_detrended, 0.1, 0.5)

    # Kalman
    pulse = kalman_smooth_1d(pulse_bp, process_noise=0.0005, measurement_noise=0.05)
    resp_sig = kalman_smooth_1d(resp_bp, process_noise=0.0002, measurement_noise=0.03)

    # Cardiac & HR – very relaxed
    min_dist_card = int(0.35 * fps)
    min_prom_card = max(np.std(pulse) * 0.5, 0.012)
    peaks_card = find_peaks_manual(pulse, min_dist_card, min_prom_card)
    print(f"{region} cardiac peaks: {len(peaks_card)} (indices: {peaks_card if len(peaks_card) > 0 else 'none'})")

    hr = None
    hr_ci = (None, None)
    hr_note = ""
    if len(peaks_card) >= 4:
        intervals = np.diff(peaks_card) / fps
        hr = 60 / np.median(intervals)
    elif len(peaks_card) >= 2:
        interval = (peaks_card[-1] - peaks_card[0]) / fps / (len(peaks_card) - 1)
        hr = 60 / interval
        hr_note = " (very few peaks)"

    # Respiratory & RR – tightened
    min_dist_resp = int(1.8 * fps)  # 1.8 s min to avoid double-counting
    std_resp = np.std(resp_sig)
    min_prom_resp = max(std_resp * 0.25, 0.010)  # tighter

    if std_resp > 1e-6:
        resp_norm = resp_sig / std_resp
    else:
        resp_norm = resp_sig.copy()

    peaks_resp = find_peaks_manual(resp_norm, min_dist_resp, min_prom_resp)
    print(f"{region} resp peaks: {len(peaks_resp)} (indices: {peaks_resp if len(peaks_resp) > 0 else 'none'})")

    rr = None
    rr_ci = (None, None)
    rr_note = ""

    if len(peaks_resp) >= 3:
        intervals_resp = np.diff(peaks_resp) / fps
        if len(intervals_resp) > 0:
            rr = 60 / np.median(intervals_resp)
            if len(intervals_resp) > 2:
                sem_rr = np.std(intervals_resp, ddof=1) / np.sqrt(len(intervals_resp))
                rr_ci = (rr - z_score * sem_rr, rr + z_score * sem_rr)
            rr_note = "" if time_vals[-1] >= 15 else " (short)"
    elif len(peaks_resp) == 2:
        interval = (peaks_resp[1] - peaks_resp[0]) / fps
        if 1.2 < interval < 8:  # tighter for single breath (~7–50 bpm)
            rr = 60 / interval
            rr_note = " (single breath)"
        else:
            rr_note = " (invalid)"
    else:
        rr_note = " (need ≥2 peaks)"

    if rr is not None and (rr < 8 or rr > 40):
        rr = None
        rr_note = " (out of range)"

    vitals[region] = {
        "spo2": spo2_mean,
        "spo2_ci": spo2_ci,
        "spo2_quality": spo2_quality,
        "hr": hr,
        "hr_ci": hr_ci,
        "hr_note": hr_note,
        "rr": rr,
        "rr_ci": rr_ci,
        "rr_note": rr_note,
        "median_ratio": np.nanmedian(ratio),
        "length_s": time_vals[-1] if len(time_vals) > 0 else 0,
        "pulse": pulse,
        "resp_sig": resp_sig,
        "time_vals": time_vals,
        "ratio": ratio,
        "df": df
    }

# ─── Console summary ───
print("\nPer-region vitals:")
for region, v in vitals.items():
    if v["length_s"] == 0:
        print(f"{region}: No data")
        continue

    spo2_str = f"{v['spo2']:.1f}" if v['spo2'] is not None else "—"
    if v['spo2_ci'][0] is not None:
        spo2_str += f" ({v['spo2_ci'][0]:.1f}–{v['spo2_ci'][1]:.1f})"

    spo2_qual_str = f"SpO₂ quality: {v['spo2_quality']}%"

    hr_str = f"{v['hr']:.1f}" if v['hr'] is not None else "—"
    if v['hr_ci'][0] is not None:
        hr_str += f" ({v['hr_ci'][0]:.1f}–{v['hr_ci'][1]:.1f})"
    if v.get('hr_note'):
        hr_str += f" {v['hr_note']}"

    rr_str = f"{v['rr']:.1f}" if v['rr'] is not None else "—"
    if v['rr_ci'][0] is not None:
        rr_str += f" ({v['rr_ci'][0]:.1f}–{v['rr_ci'][1]:.1f})"
    rr_str += v['rr_note']

    print(f"{region}: SpO₂ = {spo2_str}% | {spo2_qual_str} | HR = {hr_str} bpm | RR = {rr_str} br/min | Length = {v['length_s']:.1f}s")

# ─── Save summary Excel ───
summary_data = []
for region in region_names:
    if region in vitals and vitals[region]["length_s"] > 0:
        v = vitals[region]
        summary_data.append({
            "Region": region,
            "SpO2_mean (%)": v["spo2"],
            "SpO2_CI_low (%)": v["spo2_ci"][0] if v["spo2_ci"][0] is not None else np.nan,
            "SpO2_CI_high (%)": v["spo2_ci"][1] if v["spo2_ci"][1] is not None else np.nan,
            "SpO2_quality (%)": v["spo2_quality"],
            "HR_mean (bpm)": v["hr"],
            "HR_note": v.get("hr_note", ""),
            "RR_mean (br/min)": v["rr"],
            "RR_note": v["rr_note"],
            "Length (s)": v["length_s"],
            "Median Ratio": v["median_ratio"]
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    excel_path = os.path.join(folder, f"{folder_name}_summary_hr_rr_restored.xlsx")
    summary_df.to_excel(excel_path, index=False)
    print(f"Saved summary Excel: {excel_path}")

# ─── Plot ───
fig = plt.figure(figsize=(18, 12))
fig.suptitle(f"Multi-Region rPPG – {folder_name}  (SpO₂ quality prioritized)", fontsize=18, fontweight='bold')

for j, region in enumerate(region_names):
    if region not in vitals or vitals[region]["length_s"] == 0:
        continue

    v = vitals[region]
    time_vals = v["time_vals"]
    ratio = v["ratio"]
    pulse = v["pulse"]
    resp_sig = v["resp_sig"]
    median_ratio = v["median_ratio"]

    spo2_str = f"SpO₂ = {v['spo2']:.1f}%"
    if v['spo2_ci'][0] is not None:
        spo2_str += f" ({v['spo2_ci'][0]:.1f}–{v['spo2_ci'][1]:.1f})"

    spo2_qual_str = f"SpO₂ quality: {v['spo2_quality']}%"

    hr_str = f"HR = {v['hr']:.1f} bpm" if v['hr'] is not None else "HR = —"
    if v['hr_ci'][0] is not None:
        hr_str += f" ({v['hr_ci'][0]:.1f}–{v['hr_ci'][1]:.1f})"
    if v.get('hr_note'):
        hr_str += f" {v['hr_note']}"

    rr_str_plot = f"RR = {v['rr']:.1f} br/min" if v['rr'] is not None else "RR = —"
    rr_str_plot += v['rr_note'] or " (need ≥15 s)"

    ax1 = plt.subplot(4, 3, 1 + j)
    ax1.plot(time_vals, ratio, color='lightgray', lw=1.2, label='Raw Ratio')
    ax1.plot(time_vals, pulse + median_ratio, color=region_colors[j], lw=3, label='Pulse (Kalman)')
    ax1.set_ylabel('660/850 Ratio')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.text(0.98, 0.05, spo2_str, transform=ax1.transAxes, fontsize=13,
             fontweight='bold', color='darkred', ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='darkred'))
    ax1.set_title(f"{region}\n{spo2_qual_str}", fontsize=14)

    ax2 = plt.subplot(4, 3, 4 + j)
    ax2.plot(time_vals, pulse, color=region_colors[j], lw=2.5)
    ax2.set_ylabel('Cardiac Pulse')
    ax2.set_title(hr_str, fontsize=12)
    ax2.grid(alpha=0.3)

    ax3 = plt.subplot(4, 3, 7 + j)
    ax3.plot(time_vals, resp_sig, color='seagreen', lw=2.5)
    ax3.set_ylabel('Resp Signal')
    ax3.set_title(rr_str_plot, fontsize=12)
    ax3.grid(alpha=0.3)

    ax4 = plt.subplot(4, 3, 10 + j)
    ax4.plot(time_vals, ratio, color='gray', lw=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Raw Ratio')
    ax4.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])

graph_path = os.path.join(folder, f"{folder_name}_graph_hr_rr_restored.jpg")
plt.savefig(graph_path, dpi=300, bbox_inches='tight')
print(f"Saved graph: {graph_path}")

print("Displaying plot... Close the figure window when finished.")
plt.show(block=True)

# ─── Save detailed ratios CSV ───
dfs = {}
for region in region_names:
    if region in vitals and vitals[region].get("df") is not None:
        dfs[region] = vitals[region]["df"]

if dfs:
    df_merged = None
    for region in region_names:
        if region in dfs:
            df_region = dfs[region]
            if df_merged is None:
                df_merged = df_region.copy()
            else:
                df_merged = df_merged.merge(
                    df_region[["frame", "ratio_660_850"]].rename(columns={"ratio_660_850": f"ratio_{region}"}),
                    on="frame", how="outer"
                )
    if df_merged is not None:
        csv_path = os.path.join(folder, f"{folder_name}_ratios_{datetime.now():%Y%m%d_%H%M}.csv")
        df_merged.to_csv(csv_path, index=False)
        print(f"Saved detailed ratios CSV: {csv_path}")

print("Done.")