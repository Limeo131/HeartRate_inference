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




def FeatureMap2Heatmap( x, Score1, Score2, Score3):
    ## initial images
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
    VIPL_test_list = args.csv_info #'/content/drive/MyDrive/facemedAI/phys/VIPL_fold1_test1.txt'
    model_path = args.model_path #'/content/drive/MyDrive/facemedAI/phys/Physformer_VIPL_fold1.pkl'

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


    for idx, row in df.iterrows():
        key = str(row['key']).strip()

        if 'source1' not in key and 'source2' not in key and 'source3' not in key:
            continue

        # if 'source1' not in key or 'v1' not in key or 'p10' not in key:
        #     continue

        video_name = key + ".avi"
        video_path = os.path.join(args.video_dir, video_name)
        if not os.path.exists(video_path):
            continue

        frame_cnt = int(row['frame_cnt'])
        fps = float(row['fps'])
        gt_hr = float(row['hr_mean'])

        print(f"\n[INFO] Processing {video_name} ({idx+1}/{len(df)})")
        log_file.write(f"\n[INFO] Processing {video_name}\n")
        log_file.flush()

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

        results_rPPG = []
        results_HR_pred = []
        rPPG_long = []
        with torch.no_grad():
            for i, sample_batched in enumerate(dataloader_test):

                inputs = sample_batched['video_x'].to(device)
                clip_average_HR, frame_rate = sample_batched['clip_average_HR_peaks'].to(device), sample_batched['framerate'].to(device)

                rPPG_long = torch.randn(1).to(device)

                HR = 0.0
                for clip in range(inputs.shape[1]):
                    rPPG, Score1, Score2, Score3 = model(inputs[:,clip,:,:,:,:], gra_sharp)
                    rPPG = rPPG[0, 30:30+160]

                    HR_predicted = TorchLossComputer.cross_entropy_power_spectrum_forward_pred(rPPG, frame_rate)+40
                    HR += HR_predicted

                    print(HR_predicted)

                    rPPG_long = torch.cat((rPPG_long, rPPG),dim=0)

                HR = HR/inputs.shape[1]

                log_file.write('\n sample number :%d \n' % (i+1))
                log_file.write('\n')
                log_file.flush()

                visual = FeatureMap2Heatmap(inputs[:,inputs.shape[1]-1,:,:,:,:], Score1, Score2, Score3)

                ## save the results as .mat
                #results_rPPG.append({'file_name': video_name, 'rPPG': rPPG_long[1:].cpu().data.numpy()})
                results_HR_pred.append({'file_name': video_name, 'HR': HR})
                print(f"[INFO] HR: {HR} | GT: {gt_hr}")
                log_file.write(f"[INFO] HR: {HR} | GT: {gt_hr}\n")
                log_file.flush()

        shutil.rmtree(save_dir)
        print(f"[INFO] Finished processing {video_name}")
        log_file.write(f"[INFO] Finished processing {video_name}\n")
        log_file.flush()


    # visual and save
    #visual = FeatureMaP2Heatmap(x_visual, x_visual3232, x_visual1616)
    sio.savemat( args.log+'/'+'outputs_rPPG_concat.mat' , {'outputs_rPPG_concat': results_rPPG})
    #sio.savemat( args.log+'/'+args.log+ '_HR.mat' , {'outputs_HR': results_HR_pred})

    print('Finished val')
    log_file.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default="/home/siming/physformer_vipl")
    parser.add_argument('--model_path', type=str, default="/home/siming/physformer_vipl/Physformer_VIPL_fold1.pkl")
    parser.add_argument('--video_dir', type=str, default="/mnt/vdb/vipl")
    parser.add_argument('--csv_info', type=str, default="/mnt/vdb/vipl/vipl_sample_info.csv")
    parser.add_argument('--log', type=str, default="/home/siming/physformer_vipl/Inference_Physformer_VIPL")
    parser.add_argument('--clip_size', type=int, default=160)
    parser.add_argument('--clip_overlap', type=int, default=60)
    args = parser.parse_args()

    train_test()
