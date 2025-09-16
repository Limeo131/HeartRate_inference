import os
import sys
import cv2
import shutil
import pandas as pd
import math
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.signal import windows
from model.physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from loadtemporal_data_test import Normaliztion, ToTensor, VIPL
from torchlosscomputer import TorchLossComputer
import json

from scipy.signal import butter, filtfilt, detrend, welch


# def estimate_resp_rate(rppg: np.ndarray, fs: float,
#                        hz_band=(0.08, 0.50),
#                        nperseg_sec=20.0,
#                        min_peak_rel=2) -> float:
#     """
#     Estimate respiration rate (breaths/min) from the entire rPPG segment.
#     Returns np.nan if estimation is unstable.
#     """
#     ...


from scipy.signal import butter, filtfilt, detrend, welch

# def estimate_resp_rate_psd(
#     rppg: np.ndarray,
#     fs: float,
#     band=(0.10, 0.50),     # 6–30 breaths/min
#     nperseg_sec=20.0,      # Welch segment length in seconds
#     snr_thresh=3.0,        # peak / median power threshold (lower → more permissive)
#     return_debug=False
# ):
#     """
#     Estimate respiration rate (breaths/min) from a single rPPG segment using PSD peak.
#     Returns: rr_bpm (float) or (rr_bpm, debug_dict) if return_debug=True.
#     """
#     ...


def estimate_resp_rate_psd(rppg, fs, band=(0.10, 0.50), nperseg_sec=20.0, return_snr=True):
    x = detrend(np.asarray(rppg, dtype=float))
    lo, hi = band
    b, a = butter(3, [lo/(fs/2), hi/(fs/2)], btype="band")
    xf = filtfilt(b, a, x)

    nperseg = int(min(len(xf), max(4*fs, nperseg_sec*fs)))
    noverlap = int(0.5*nperseg)
    f, Pxx = welch(xf, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False)

    mask = (f >= lo) & (f <= hi)
    f_band, P_band = f[mask], Pxx[mask]
    if P_band.size == 0:
        return (np.nan, np.nan) if return_snr else np.nan

    i = int(np.argmax(P_band))
    rr_bpm = float(f_band[i] * 60.0)
    snr = float(P_band[i] / (np.median(P_band) + 1e-12))
    return (rr_bpm, snr) if return_snr else rr_bpm



def video2frames(video_path, save_dir, target_frame_count, log_file=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0 or not cap.isOpened():
        msg = f"[SKIP] Cannot read frames: {video_path}"
        print(msg)
        if log_file: log_file.write(msg + '\n')
        return False

    os.makedirs(save_dir, exist_ok=True)

    step = total_frames / target_frame_count
    for i in range(target_frame_count):
        frame_idx = round(i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Could not read frame {frame_idx}")
            continue
        frame = cv2.resize(frame, (132, 132))[2:130, 2:130, :]
        save_path = os.path.join(save_dir, f"image_{i+1:05d}.png")
        cv2.imwrite(save_path, frame)

    cap.release()
    msg = f"[INFO] Saved {target_frame_count} frames to {save_dir}"
    print(msg)
    if log_file: log_file.write(msg + '\n')
    return True

from scipy.signal import butter, filtfilt, find_peaks

def bandpass_filter(x, fs, lo=0.7, hi=3.0, order=3):
    # 0.7–3.0 Hz ≈ 42–180 bpm
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def compute_rr_and_sdnn(rppg, fs):
    """
    rppg: 1D numpy array
    fs: sampling rate (float)
    returns: rr_ms (np.ndarray), sdnn (float), peaks (np.ndarray)
    """
    if len(rppg) < int(fs*3):  # Too short to stably compute RR/SDNN
        return np.array([]), np.nan, np.array([])
    x = bandpass_filter(rppg, fs)
    # Peak distance ≥ 0.3s to suppress dense false peaks; prominence adaptive to signal energy
    distance = max(1, int(0.3 * fs))
    prominence = max(1e-6, np.std(x) * 0.2)
    peaks, _ = find_peaks(x, distance=distance, prominence=prominence)
    if len(peaks) < 2:
        return np.array([]), np.nan, peaks
    t = np.arange(len(x)) / fs
    rr_ms = np.diff(t[peaks]) * 1000.0
    # Clean based on physiological range
    rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
    sdnn = rr_ms.std(ddof=1) if rr_ms.size > 1 else np.nan
    return rr_ms, sdnn, peaks



def FeatureMap2Heatmap( x, Score1, Score2, Score3):
    ## initial images
    org_img = x[0,:,32,:,:].cpu()
    org_img = org_img.data.numpy()*128+127.5
    org_img = org_img.transpose((1, 2, 0))
    #org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log+'/'+args.log + 'visual.jpg', org_img)

    # [B, head, 640, 640]
    org_img = Score1[0, 1].cpu().data.numpy()*4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log+'/'+'Score1_head1.jpg', org_img)

    org_img = Score2[0, 1].cpu().data.numpy()*4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log+'/'+'Score2_head1.jpg', org_img)

    org_img = Score3[0, 1].cpu().data.numpy()*4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log+'/'+'Score3_head1.jpg', org_img)


# main function
def train_test():

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ 'inference_log.txt', 'w')

    VIPL_root_list = args.log + '/VIPL_frames/'
    VIPL_test_list = args.csv_info
    model_path = args.model_path

    print('evaluation~ forward!\n')

    gra_sharp = 2.0
    device = torch.device("cpu")
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,128,128), patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    df = pd.read_csv(args.csv_info)
    df.columns = df.columns.str.strip()

    output_frame_dir = os.path.join(args.log, "VIPL_frames")
    os.makedirs(output_frame_dir, exist_ok=True)

    results_rPPG_all = []
    results_HR_pred_all = []
    results_RR_SDNN_all = []

    processed_count = 0
    MAX_N = 490  # Only run inference on the first 490 samples

    for idx, row in df.iterrows():
        key = str(row['key']).strip()

        video_name = key + ".avi"
        video_path = os.path.join(args.video_dir, video_name)
        if not os.path.exists(video_path):
            continue

        processed_count += 1
        frame_cnt = int(row['frame_cnt'])
        fps = float(row['fps'])
        gt_hr = float(row['hr_mean'])

        print(f"\n[INFO] Processing {video_name} ({idx+1}/{len(df)})")
        log_file.write(f"\n[INFO] Processing {video_name}\n")
        log_file.flush()

        if processed_count > MAX_N:
            break

        # === Step 1: Video to frames ===
        save_dir = os.path.join(output_frame_dir, key)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        ok = video2frames(video_path, save_dir, target_frame_count=frame_cnt, log_file=log_file)
        if not ok:
            continue

        # === Step 2: DataLoader ===
        VIPL_testDL = VIPL(VIPL_test_list, VIPL_root_list, transform=transforms.Compose([Normaliztion(), ToTensor()]), single_file=key)
        dataloader_test = DataLoader(VIPL_testDL, batch_size=1, shuffle=False, num_workers=2)

        with torch.no_grad():
            for i, sample_batched in enumerate(dataloader_test):

                inputs = sample_batched['video_x'].to(device)
                frame_rate = float(sample_batched['framerate'].item())  # NEW: sampling rate

                if inputs.shape[1] == 0:  # No clip available, skip directly
                    print(f"[WARN] No valid clips for {video_name}, skipping.")
                    log_file.write(f"[WARN] No valid clips for {video_name}, skipping.\n")
                    continue

                # NEW: Use a list to collect rPPG from each clip (CPU numpy)
                rppg_segments = []
                HR = 0.0

                for clip in range(inputs.shape[1]):
                    rPPG, Score1, Score2, Score3 = model(inputs[:, clip, :, :, :, :], gra_sharp)
                    # Extract the same valid segment as original code
                    rPPG = rPPG[0, 30:30+160].detach().cpu().numpy()
                    rppg_segments.append(rPPG)

                    # Original HR spectrum peak method
                    HR_predicted = TorchLossComputer.cross_entropy_power_spectrum_forward_pred(
                        torch.from_numpy(rPPG).to(device), torch.tensor(frame_rate).to(device)
                    ) + 40
                    HR += float(HR_predicted)

                HR = HR / inputs.shape[1]

                # === NEW: Concatenate full rPPG and compute RR and SDNN ===
                rppg_long = np.concatenate(rppg_segments, axis=0) if len(rppg_segments) > 0 else np.array([])
                rr_ms, sdnn, peaks = compute_rr_and_sdnn(rppg_long, frame_rate)
                # --- NEW: Estimate full-segment respiration rate (breaths/min) + SNR
                resp_bpm, resp_snr = estimate_resp_rate_psd(rppg_long, frame_rate, return_snr=True)

                # Ensure scalar/NaN
                resp_bpm = float(resp_bpm) if isinstance(resp_bpm, (int, float, np.floating)) and np.isfinite(resp_bpm) else np.nan
                resp_snr = float(resp_snr) if isinstance(resp_snr, (int, float, np.floating)) and np.isfinite(resp_snr) else np.nan

                log_file.write(f"[INFO] RespRate: {'NaN' if np.isnan(resp_bpm) else f'{resp_bpm:.2f}'} bpm | SNR: {'NaN' if np.isnan(resp_snr) else f'{resp_snr:.2f}'}\n")
                print(f"[INFO] RespRate: {'NaN' if np.isnan(resp_bpm) else f'{resp_bpm:.2f}'} bpm | SNR: {'NaN' if np.isnan(resp_snr) else f'{resp_snr:.2f}'}")

                # Log
                log_file.write('\n sample number :%d \n' % (i+1))
                log_file.write(f"[INFO] HR: {HR:.2f} | GT: {gt_hr:.2f} | fs: {frame_rate:.2f} Hz | SDNN: {sdnn if not np.isnan(sdnn) else 'NaN'} ms | RR_n={len(rr_ms)}\n")
                log_file.flush()
                print(f"[INFO] HR: {HR:.2f} | GT: {gt_hr:.2f} | fs: {frame_rate:.2f} Hz | SDNN: {sdnn if not np.isnan(sdnn) else 'NaN'} ms | RR_n={len(rr_ms)}")

                # Visualization (keep original logic)
                visual = FeatureMap2Heatmap(inputs[:, inputs.shape[1]-1, :, :, :, :], Score1, Score2, Score3)

                results_rPPG_all.append({
                    'file_name': video_name,
                    'rPPG': rppg_long,
                    'fs': frame_rate
                })
                results_HR_pred_all.append({'file_name': video_name, 'HR': HR, 'HR_gt': gt_hr})
                results_RR_SDNN_all.append({
                    'file_name': video_name,
                    'sdnn_ms': float(sdnn) if not np.isnan(sdnn) else np.nan,
                    'rr_ms': rr_ms.astype(float),
                    'resp_bpm': resp_bpm,        # Already scalar or NaN
                    'resp_snr': resp_snr         # NEW: Respiration peak SNR
                })

        shutil.rmtree(save_dir)
        print(f"[INFO] Finished processing {video_name}")
        log_file.write(f"[INFO] Finished processing {video_name}\n")
        log_file.flush()

    # === Save MAT: containing rPPG, RR, SDNN ===
    # Note: MAT has weaker compatibility for variable-length lists than npy/csv; saved here as cell-like structure
    sio.savemat(os.path.join(args.log, 'outputs_rPPG_rr_sdnn.mat'), {
        'results_rPPG': np.array(results_rPPG_all, dtype=object),
        'results_HR_pred': np.array(results_HR_pred_all, dtype=object),
        'results_RR_SDNN': np.array(results_RR_SDNN_all, dtype=object),
    }, do_compression=True)

    # --- build maps for CSV ---
    hr_map = {d['file_name']: float(d['HR']) for d in results_HR_pred_all}
    hr_gt_map = {d['file_name']: float(d.get('HR_gt', float('nan'))) for d in results_HR_pred_all}  # NEW
    fs_map = {d['file_name']: float(d.get('fs', np.nan)) for d in results_rPPG_all}

    rr_map = {}
    for d in results_RR_SDNN_all:
        rr_vals = d['rr_ms']
        rr_map[d['file_name']] = {
            'sdnn_ms': float(d['sdnn_ms']) if d['sdnn_ms'] == d['sdnn_ms'] else np.nan,
            'rr_ms': rr_vals,
            'resp_bpm': float(d.get('resp_bpm', np.nan)) if d.get('resp_bpm', None) is not None and np.isfinite(d.get('resp_bpm')) else np.nan,
            'resp_snr': float(d.get('resp_snr', np.nan)) if d.get('resp_snr', None) is not None and np.isfinite(d.get('resp_snr')) else np.nan,
        }

    all_keys = sorted(set(hr_map) | set(rr_map) | set(fs_map))

    rows = []
    for k in all_keys:
        rr_vals = rr_map.get(k, {}).get('rr_ms', np.array([], dtype=float))
        sdnn = rr_map.get(k, {}).get('sdnn_ms', np.nan)
        fps  = fs_map.get(k, np.nan)

        rows.append({
            'file_name': k,
            'fps': fps,
            'HR_pred': hr_map.get(k, np.nan),
            'HR_gt': hr_gt_map.get(k, np.nan),                 # NEW: ground truth HR
            # 'HR_abs_err': (abs(hr_map.get(k, np.nan) - hr_gt_map.get(k, np.nan))
            #                if np.isfinite(hr_map.get(k, np.nan)) and np.isfinite(hr_gt_map.get(k, np.nan))
            #                else np.nan),                      # Optional: error
            'SDNN_ms': sdnn,
            'RR_count': int(len(rr_vals)),
            'RR': json.dumps(rr_vals.tolist() if isinstance(rr_vals, np.ndarray) else list(rr_vals)),
            'RespRate_bpm': rr_map.get(k, {}).get('resp_bpm', np.nan),
            'RespRate_SNR': rr_map.get(k, {}).get('resp_snr', np.nan),
        })

    out_csv = os.path.join(args.log, 'all_metrics_with_rr.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[INFO] Saved {out_csv} with {len(rows)} rows")

    print('Finished val')
    log_file.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default="/home/siming/physformer_vipl")
    parser.add_argument('--model_path', type=str, default="/home/siming/physformer_vipl/Physformer_VIPL_fold1.pkl")
    parser.add_argument('--video_dir', type=str, default="/mnt/vdb/vipl")
    parser.add_argument('--csv_info', type=str, default="/mnt/vdb/vipl/vipl_sample_info.csv")
    parser.add_argument('--log', type=str, default="/home/siming/physformer_vipl/Inference_Physformer_VIPL_rrhrv")
    parser.add_argument('--clip_size', type=int, default=160)
    parser.add_argument('--clip_overlap', type=int, default=60)
    args = parser.parse_args()

    train_test()
