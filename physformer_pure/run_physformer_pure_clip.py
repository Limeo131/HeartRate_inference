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
from loadtemporal_data_test_clip import Normaliztion, ToTensor, VIPL
from torchlosscomputer import TorchLossComputer


def FeatureMap2Heatmap(x, Score1, Score2, Score3):
    """
    Save intermediate feature maps and attention maps as images.
    Mainly used for visual debugging of the model.
    """
    # Take one slice of the input tensor and rescale to an image
    org_img = x[0, :, 32, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    cv2.imwrite(args.log + '/' + args.log + 'visual.jpg', org_img)

    # Save Score1 heatmap
    org_img = Score1[0, 1].cpu().data.numpy() * 4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log + '/' + 'Score1_head1.jpg', org_img)

    # Save Score2 heatmap
    org_img = Score2[0, 1].cpu().data.numpy() * 4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log + '/' + 'Score2_head1.jpg', org_img)

    # Save Score3 heatmap
    org_img = Score3[0, 1].cpu().data.numpy() * 4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log + '/' + 'Score3_head1.jpg', org_img)


def train_test():
    """
    Main evaluation loop:
    - Load PhysFormer model
    - Iterate over JSON metadata files (each video)
    - Run inference clip-by-clip
    - Save predictions and ground truth HR to CSV
    """
    # Create log directory if it doesn’t exist
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    log_file = open(args.log + '/' + 'inference_log.txt', 'w')

    # Path to pretrained model weights
    model_path = args.model_path

    print('evaluation~ forward!\n')

    # Load PhysFormer model on CPU
    device = torch.device("cpu")
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(160, 128, 128), patches=(4, 4, 4),
        dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7
    )
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Collect all JSON metadata files in the dataset directory
    json_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".json")])

    all_results = []

    # Loop through each JSON (each video)
    for idx, jf in enumerate(json_files):
        json_path = os.path.join(args.video_dir, jf)
        key = jf.replace(".json", "")
        print(f"\n[INFO] Processing {key} ({idx+1}/{len(json_files)})")

        # Construct dataset from JSON (PURE dataset format)
        dataset = VIPL(
            info_list=[json_path],
            root_dir=args.video_dir,
            transform=transforms.Compose([Normaliztion(), ToTensor()])
        )

        dataloader_test = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        # Inference loop (no gradient computation needed)
        with torch.no_grad():
            for i, sample in enumerate(dataloader_test):

                # Input tensor shape: [1, clip_num, C, D, H, W]
                inputs = sample['video_x'].to(device)
                frame_rate = sample['framerate'].item()  # scalar FPS
                clip_gt_HRs = sample['clip_average_HR_peaks'][0].tolist()  # list of ground-truth HRs

                pred_HRs = []

                # Run model for each clip
                for clip_idx in range(inputs.shape[1]):
                    rPPG, _, _, _ = model(inputs[:, clip_idx, :, :, :, :], gra_sharp=2.0)
                    # Select central segment of rPPG output
                    rPPG = rPPG[0, 30:30 + 160]
                    # Estimate HR from rPPG spectrum (plus 40 offset for calibration)
                    HR_pred = TorchLossComputer.cross_entropy_power_spectrum_forward_pred(rPPG, frame_rate) + 40
                    pred_HRs.append(HR_pred.item())

                # Log and save predictions
                for clip_idx, (pred, gt) in enumerate(zip(pred_HRs, clip_gt_HRs)):
                    all_results.append({
                        'video': key,
                        'clip_idx': clip_idx,
                        'pred_HR': pred,
                        'gt_HR': gt
                    })
                    msg = f"[INFO] {key} | Clip {clip_idx} | Pred: {pred:.2f} | GT: {gt:.2f}"
                    print(msg)
                    log_file.write(msg + "\n")

        # Save all accumulated results to CSV (updates after each video)
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(args.log, "clip_HR_results.csv"), index=False)
        print(f"[INFO] Saved clip HR results to {args.log}/clip_HR_results.csv")

    log_file.close()


if __name__ == "__main__":
    # Argument parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default="/home/siming/physformer_pure")
    parser.add_argument('--model_path', type=str, default="/home/siming/physformer_pure/Physformer_VIPL_fold1.pkl")
    parser.add_argument('--video_dir', type=str, default="/mnt/vdb/pure")  # Directory with JSON files + frames
    parser.add_argument('--csv_info', type=str, default="/home/siming/physformer_pure/pure_full_info.csv")
    parser.add_argument('--log', type=str, default="/home/siming/physformer_pure/Inference_PURE_JSON")
    parser.add_argument('--clip_size', type=int, default=160)
    parser.add_argument('--clip_overlap', type=int, default=60)
    args = parser.parse_args()

    # Run evaluation
    train_test()
