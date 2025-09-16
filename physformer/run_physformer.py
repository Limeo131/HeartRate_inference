# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# import cv2
# import shutil
# import pandas as pd
# import math
# import torch
# import numpy as np
# import scipy.io as sio
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from scipy.signal import windows

# #from physformer.model.physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
# from model.physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
# from loadtemporal_data_test import Normaliztion, ToTensor, VIPL



# def video2frames(video_name, video_dir, save_dir, csv_fps, csv_frame_cnt, target_fps=30, log_file=None):
#     video_path = os.path.join(video_dir, video_name)
#     cap = cv2.VideoCapture(video_path)
#     original_fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     if total_frames == 0:
#         msg = f"[SKIP] Cannot read frames for: {video_name}"
#         print(msg)
#         if log_file: log_file.write(msg + '\n')
#         return False

#     if abs(total_frames - csv_frame_cnt) > 2:
#         msg = f"[WARNING] Frame count mismatch: CSV={csv_frame_cnt}, Actual={total_frames}"
#         print(msg)
#         if log_file: log_file.write(msg + '\n')

#     os.makedirs(save_dir, exist_ok=True)
#     step = original_fps / target_fps

#     frame_idx = 0
#     save_idx = 0

#     while True:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.resize(frame, (132, 132))[2:130, 2:130, :]
#         save_name = f"image_{save_idx:05d}.png"
#         save_path = os.path.join(save_dir, save_name)
#         cv2.imwrite(save_path, frame)

#         frame_idx += step
#         save_idx += 1
#         if int(frame_idx) >= total_frames:
#             break

#     cap.release()
#     msg = f"[INFO] Saved {save_idx} frames to {save_dir}"
#     print(msg)
#     if log_file: log_file.write(msg + '\n')
#     return True


# def overlap_add(prev, new, overlap_len):
#     if overlap_len == 0:
#         return np.concatenate([prev, new])

#     left = prev[-overlap_len:]
#     right = new[:overlap_len]

#     w = windows.hann(overlap_len * 2)
#     left_w = w[:overlap_len]
#     right_w = w[overlap_len:]

#     blended = left * left_w + right * right_w

#     merged = np.concatenate([prev[:-overlap_len], blended, new[overlap_len:]])
#     return merged


# def run_inference(args):
#     os.makedirs(args.log, exist_ok=True)
#     log_file = open(os.path.join(args.log, "inference_log.txt"), "w")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     msg = f"[INFO] Using device: {device}"
#     print(msg)
#     log_file.write(msg + '\n')
#     log_file.flush()

#     video_dir = args.video_dir
#     csv_info = args.csv_info
#     output_root = os.path.join(args.input_data, 'VIPL_frames')
#     model_path = os.path.join(args.input_data, 'Physformer_VIPL_fold1.pkl')

#     model = ViT_ST_ST_Compact3_TDC_gra_sharp(
#         image_size=(160,128,128),
#         patches=(4,4,4),
#         dim=96, ff_dim=144, num_heads=4, num_layers=12,
#         dropout_rate=0.1, theta=0.7
#     )
#     model = model.to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     gra_sharp = 2.0

#     df = pd.read_csv(csv_info)
#     df.columns = df.columns.str.strip()

#     results_rPPG = []
#     results_HR = []

#     all_videos = df['file_name'].unique().tolist()
#     total_videos = len(all_videos)
#     video_count = 0

#     for idx, row in df.iterrows():
#         video_name = str(row['file_name']).strip()
#         if not video_name.endswith('.mp4'):
#             video_name += '.mp4'

#         fps_val = row['fps']
#         frame_cnt_val = row['frame_cnt']
#         gt_hr = float(row['hr_true'])

#         if pd.isna(fps_val) or math.isnan(fps_val) or pd.isna(frame_cnt_val) or math.isnan(frame_cnt_val):
#             continue

#         csv_fps = float(fps_val)
#         csv_frame_cnt = int(frame_cnt_val)
#         video_path = os.path.join(video_dir, video_name)
#         if not os.path.exists(video_path):
#             continue

#         msg = f"\n[INFO] Processing {video_name} ({video_count+1}/{total_videos})"
#         print(msg)
#         log_file.write(msg + '\n')
#         log_file.flush()

#         # === Step 1: 转帧 ===
#         save_dir = os.path.join(output_root, video_name.replace('.mp4',''))
#         if os.path.exists(save_dir):
#             shutil.rmtree(save_dir)
#         ok = video2frames(video_name, video_dir, save_dir, csv_fps, csv_frame_cnt, target_fps=30, log_file=log_file)
#         if not ok:
#             continue

#         # === Step 2: Dataset & DataLoader ===
#         dataset = VIPL(
#             info_list=csv_info,
#             root_dir=output_root,
#             transform=transforms.Compose([Normaliztion(), ToTensor()]),
#             clip_size=args.clip_size,
#             clip_overlap=args.clip_overlap,
#             single_file=video_name
#         )
#         dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#         rPPG_long = []
#         framerate = 30.0

#         for i, batch in enumerate(dataloader):
#             inputs = batch['video_x'].to(device).permute(0,2,1,3,4)
#             rPPG, _, _, _ = model(inputs, gra_sharp)
#             rPPG_clip = rPPG[0].detach().cpu().numpy()
#             if len(rPPG_long) == 0:
#                 rPPG_long = rPPG_clip
#             else:
#                 rPPG_long = overlap_add(rPPG_long, rPPG_clip, args.clip_overlap)

#         # freqs = np.fft.rfftfreq(len(rPPG_long), d=1/framerate)
#         # fft = np.abs(np.fft.rfft(rPPG_long))
#         # peak_freq = freqs[np.argmax(fft[1:])+1]
#         # pred_HR = peak_freq * 60

#         ###################################################################
#         # === Better HR estimation with parabolic interpolation ===
#         freqs = np.fft.rfftfreq(len(rPPG_long), d=1/framerate)
#         fft = np.abs(np.fft.rfft(rPPG_long))

#         # Find peak index (excluding DC component)
#         peak_idx = np.argmax(fft[1:]) + 1  # +1 to correct for [1:]

#         # If peak is not at boundary (to allow interpolation)
#         if 1 <= peak_idx < len(fft) - 1:
#             y0 = fft[peak_idx - 1]
#             y1 = fft[peak_idx]
#             y2 = fft[peak_idx + 1]

#             # Parabolic interpolation offset
#             denom = y0 - 2*y1 + y2
#             if denom != 0:
#                 offset = 0.5 * (y0 - y2) / denom
#             else:
#                 offset = 0  # fallback to center

#             # Interpolated frequency
#             interp_freq = freqs[peak_idx] + offset * (freqs[1] - freqs[0])
#         else:
#             # fallback to original
#             interp_freq = freqs[peak_idx]

#         pred_HR = interp_freq * 60
#         ###################################################################

#         results_rPPG.append(rPPG_long)
#         results_HR.append({'file_name': video_name, 'pred_HR': pred_HR, 'gt_HR': gt_hr})

#         video_count += 1
#         msg = f"[INFO] Finished {video_name} ({video_count}/{total_videos}): Pred HR = {pred_HR:.2f} | GT HR = {gt_hr:.2f}"
#         print(msg)
#         log_file.write(msg + '\n')
#         log_file.flush()

#         # === Step 3: 删除帧 ===
#         shutil.rmtree(save_dir)
#         msg = f"[INFO] Deleted frames: {save_dir}"
#         print(msg)
#         log_file.write(msg + '\n')
#         log_file.flush()

#     results_rPPG_arr = np.empty(len(results_rPPG), dtype=object)
#     results_rPPG_arr[:] = results_rPPG
#     sio.savemat(f"{args.log}/outputs_rPPG_concat.mat", {'outputs_rPPG': results_rPPG_arr})
#     pd.DataFrame(results_HR).to_csv(f"{args.log}/HR_results.csv", index=False)

#     msg = f"[INFO] All done. Saved results to {args.log}"
#     print(msg)
#     log_file.write(msg + '\n')
#     log_file.flush()
#     log_file.close()

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_data', type=str, default="/home/siming/physformer")
#     parser.add_argument('--video_dir', type=str, default="/mnt/vdb/sample_video")
#     parser.add_argument('--csv_info', type=str, default="/home/siming/physformer/sample_info_300.csv")
#     parser.add_argument('--log', type=str, default="Inference_Physformer")
#     parser.add_argument('--clip_size', type=int, default=160)
#     parser.add_argument('--clip_overlap', type=int, default=60)
#     args, unknown = parser.parse_known_args()

#     run_inference(args)


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



    VIPL_root_list = args.log + '/sample_frames/'
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

    output_frame_dir = os.path.join(args.log, "sample_frames")
    os.makedirs(output_frame_dir, exist_ok=True)


    for idx, row in df.iterrows():
        key = str(row['file_name']).strip()

        video_name = key + ".mp4"
        video_path = os.path.join(args.video_dir, video_name)
        if not os.path.exists(video_path):
            continue

        frame_cnt = int(row['frame_cnt'])
        fps = float(row['fps'])
        gt_hr = float(row['hr_true'])

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

                if inputs.shape[1] == 0:
                    #inputs.shape[1] = 1
                    HR = HR/1
                else:
                    HR = HR/inputs.shape[1]

                log_file.write('\n sample number :%d \n' % (i+1))
                log_file.write('\n')
                log_file.flush()

                #visual = FeatureMap2Heatmap(inputs[:,inputs.shape[1]-1,:,:,:,:], Score1, Score2, Score3)

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
    parser.add_argument('--input_data', type=str, default="/home/siming/physformer")
    parser.add_argument('--model_path', type=str, default="/home/siming/physformer/Physformer_VIPL_fold1.pkl")
    parser.add_argument('--video_dir', type=str, default="/mnt/vdb/sample_video")
    parser.add_argument('--csv_info', type=str, default="/home/siming/physformer/sample_info.csv")
    parser.add_argument('--log', type=str, default="/home/siming/physformer/Inference_Physformer")
    parser.add_argument('--clip_size', type=int, default=160)
    parser.add_argument('--clip_overlap', type=int, default=60)
    args = parser.parse_args()

    train_test()