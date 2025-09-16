# run_physformer_pure_clip_withpre.py
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from model.physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from loadtemporal_data_test_clip import Normaliztion, ToTensor, VIPL
from torchlosscomputer import TorchLossComputer

from face_preprocess import json2frames_preprocess  # ★ Frame preprocessing with face detection/cropping

def FeatureMap2Heatmap(x, Score1, Score2, Score3):
    """
    Save intermediate feature maps and attention score maps as images.
    Useful for visual debugging of the model’s learned representations.
    """
    org_img = x[0,:,32,:,:].cpu().data.numpy()*128+127.5
    org_img = org_img.transpose((1, 2, 0))
    cv2.imwrite(os.path.join(args.log, args.log + 'visual.jpg'), org_img)

    for i, score in enumerate([Score1, Score2, Score3], 1):
        img = (score[0, 1].cpu().data.numpy()*4000).astype(np.float32)
        cv2.imwrite(os.path.join(args.log, f'Score{i}_head1.jpg'), cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

def train_test():
    """
    Main inference loop for PhysFormer on PURE dataset JSON metadata files.
    Each JSON is preprocessed into aligned face frames, then fed into the model
    clip-by-clip to predict HR.
    """
    os.makedirs(args.log, exist_ok=True)
    log_file = open(os.path.join(args.log, 'inference_log.txt'), 'w')

    # --- Load model on CPU (default setup) ---
    device = torch.device("cpu")
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,128,128), patches=(4,4,4),
                                             dim=96, ff_dim=144, num_heads=4, num_layers=12,
                                             dropout_rate=0.1, theta=0.7).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- Collect JSON files (each JSON corresponds to one video) ---
    json_files = sorted([os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(".json")])

    # (Optional) You could restrict JSONs to a whitelist and keep custom ordering
    # Example: ALLOWED_KEYS = ["07-05","07-06", ...]
    # Filter + reorder based on whitelist

    # --- Directories for outputs ---
    pre_root = os.path.join(args.log, "PURE_pre_frames")  # ★ Preprocessed frames root (Dataset root_dir)
    comp_dir = os.path.join(args.log, "comparisons")      # For saving face preprocessing comparisons
    os.makedirs(pre_root, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    saved_comp = 0; COMP_LIMIT = 100

    all_results = []
    print('evaluation~ forward!\n')

    # --- Loop over JSON metadata files ---
    for idx, json_path in enumerate(json_files):
        key = os.path.splitext(os.path.basename(json_path))[0]
        print(f"\n[INFO] Processing {key} ({idx+1}/{len(json_files)})")
        log_file.write(f"\n[INFO] Processing {key}\n"); log_file.flush()

        # === Step 1: Preprocess frames (face crop, resize, alignment) ===
        save_compare = (saved_comp < COMP_LIMIT)
        ok = json2frames_preprocess(
            json_path=json_path,
            orig_root=args.video_dir,
            out_root=pre_root,
            use_face_preprocess=True,
            out_size=128,
            preprocess_kwargs={"crop_scale": 4.2, "shift_y": 0.30, "min_box": 112},
            comparison_out_dir=comp_dir,
            save_comparison=save_compare,
            comparison_pick="middle",
            comparison_scale=2
        )
        if save_compare: saved_comp += 1
        if not ok:
            log_file.write(f"[WARN] Skip {key}: preprocess failed or no frames\n"); log_file.flush()
            continue

        # === Step 2: Build dataset/dataloader using preprocessed frames ===
        dataset = VIPL(info_list=[json_path], root_dir=pre_root,
                       transform=transforms.Compose([Normaliztion(), ToTensor()]))
        dataloader_test = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        # === Step 3: Run inference on each clip ===
        with torch.no_grad():
            for i, sample in enumerate(dataloader_test):
                inputs = sample['video_x'].to(device)      # Shape: [1, clip_num, C, D, H, W]
                frame_rate = sample['framerate'].item()
                clip_gt_HRs = sample['clip_average_HR_peaks'][0].tolist()

                pred_HRs = []
                for clip_idx in range(inputs.shape[1]):
                    rPPG, _, _, _ = model(inputs[:, clip_idx], gra_sharp=2.0)
                    rPPG = rPPG[0, 30:30+160]  # Crop to central segment
                    HR_pred = TorchLossComputer.cross_entropy_power_spectrum_forward_pred(rPPG, frame_rate) + 40
                    pred_HRs.append(HR_pred.item())

                # === Step 4: Log and save results for each clip ===
                for clip_idx, (pred, gt) in enumerate(zip(pred_HRs, clip_gt_HRs)):
                    all_results.append({'video': key, 'clip_idx': clip_idx, 'pred_HR': pred, 'gt_HR': gt})
                    msg = f"[INFO] {key} | Clip {clip_idx} | Pred: {pred:.2f} | GT: {gt:.2f}"
                    print(msg); log_file.write(msg + "\n")

        # === Step 5: Save partial results (checkpoint-friendly) ===
        df = pd.DataFrame(all_results)
        out_csv = os.path.join(args.log, "clip_HR_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Saved clip HR results to {out_csv}")
        log_file.flush()

    log_file.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default="/home/siming/physformer_pure")
    parser.add_argument('--model_path', type=str, default="/home/siming/physformer_pure/Physformer_VIPL_fold1.pkl")
    parser.add_argument('--video_dir', type=str, default="/mnt/vdb/pure")  # Contains JSON + original frames
    parser.add_argument('--log', type=str, default="/home/siming/physformer_pure/Inference_PURE_JSON_withpre")
    parser.add_argument('--clip_size', type=int, default=160)
    parser.add_argument('--clip_overlap', type=int, default=60)
    args = parser.parse_args()
    train_test()
