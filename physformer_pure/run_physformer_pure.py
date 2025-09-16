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


def FeatureMap2Heatmap(x, Score1, Score2, Score3):
    ## Initialize images
    org_img = x[0, :, 32, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log + '/' + args.log + 'visual.jpg', org_img)

    # Save Score1 heatmap [B, head, 640, 640]
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


# Main function
def train_test():
    # Create log folder if it does not exist
    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log + '/' + 'inference_log.txt', 'w')

    model_path = args.model_path  # '/content/drive/MyDrive/facemedAI/phys/Physformer_VIPL_fold1.pkl'

    print('evaluation~ forward!\n')

    gra_sharp = 2.0
    device = torch.device("cpu")

    # Load PhysFormer model
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(160, 128, 128), patches=(4, 4, 4),
        dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7
    )
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get all JSON metadata files
    json_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".json")])

    results_HR_pred = []

    # Loop through all JSON files (each represents one video)
    for idx, jf in enumerate(json_files):
        json_path = os.path.join(args.video_dir, jf)
        key = jf.replace(".json", "")
        print(f"\n[INFO] Processing {key} ({idx+1}/{len(json_files)})")

        # Build dataset
        dataset = VIPL(
            info_list=[json_path], root_dir=args.video_dir,
            transform=transforms.Compose([Normaliztion(), ToTensor()])
        )

        dataloader_test = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

        with torch.no_grad():
            for i, sample_batched in enumerate(dataloader_test):
                inputs = sample_batched['video_x'].to(device)  # Input video clips
                clip_average_HR = sample_batched['clip_average_HR_peaks'].to(device)  # Ground truth HR
                frame_rate = sample_batched['framerate'].to(device)  # Frame rate (fps)

                rPPG_long = torch.randn(1).to(device)
                HR = 0.0

                # Loop through all clips in the video
                for clip in range(inputs.shape[1]):
                    rPPG, Score1, Score2, Score3 = model(inputs[:, clip, :, :, :, :], gra_sharp=2.0)
                    rPPG = rPPG[0, 30:30+160]
                    rPPG_long = torch.cat((rPPG_long, rPPG), dim=0)

                    HR_predicted = TorchLossComputer.cross_entropy_power_spectrum_forward_pred(rPPG, frame_rate) + 40
                    HR += HR_predicted

                # Average HR across clips
                HR = HR / inputs.shape[1]

                results_HR_pred.append({'file_name': key, 'HR': HR})

                # Optionally visualize feature maps
                # FeatureMap2Heatmap(inputs[:, inputs.shape[1]-1, :, :, :, :], Score1, Score2, Score3)

                print(f"[INFO] Pred HR: {HR}, GT HR: {clip_average_HR[0].cpu().data.numpy()}")
                log_file.write(f"[INFO] {key} | HR: {HR}, GT HR: {clip_average_HR[0].cpu().data.numpy()}\n")
                log_file.flush()

    # Save results to MATLAB .mat file
    sio.savemat(os.path.join(args.log, 'outputs_HR.mat'), {'outputs_HR': results_HR_pred})
    print('[INFO] Finished inference on PURE')
    log_file.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default="/home/siming/physformer_pure")
    parser.add_argument('--model_path', type=str, default="/home/siming/physformer_pure/Physformer_VIPL_fold1.pkl")
    parser.add_argument('--video_dir', type=str, default="/mnt/vdb/pure")  # Directory of JSON + frames
    parser.add_argument('--csv_info', type=str, default="/home/siming/physformer_pure/pure_full_info.csv")
    parser.add_argument('--log', type=str, default="/home/siming/physformer_pure/Inference_PURE_JSON")
    parser.add_argument('--clip_size', type=int, default=160)
    parser.add_argument('--clip_overlap', type=int, default=60)
    args = parser.parse_args()
    train_test()
