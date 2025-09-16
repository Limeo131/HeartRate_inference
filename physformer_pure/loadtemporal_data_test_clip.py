import os
import json
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

clip_frames = 160 + 60  # 220 frames, including 60-frame overlap

class Normaliztion(object):
    def __call__(self, sample):
        video_x, framerate, clip_average_HR_peaks = sample['video_x'], sample['framerate'], sample['clip_average_HR_peaks']
        new_video_x = (video_x - 127.5) / 128
        return {
            'video_x': new_video_x,
            'framerate': framerate,
            'clip_average_HR_peaks': clip_average_HR_peaks
        }

class ToTensor(object):
    def __call__(self, sample):
        video_x = sample['video_x'].transpose((0, 4, 1, 2, 3))  # clip x C x depth x H x W
        return {
            'video_x': torch.from_numpy(video_x.astype(np.float32)).float(),
            'framerate': torch.tensor(sample['framerate'], dtype=torch.float64),
            'clip_average_HR_peaks': torch.tensor(sample['clip_average_HR_peaks'], dtype=torch.float32)
        }

class VIPL(Dataset):

    def __init__(self, info_list, root_dir, transform=None):
        # if info_list.endswith(".json"):
        #     self.json_files = [os.path.basename(info_list)]
        #     self.single_file = True
        # else:
        #     self.json_files = sorted([f for f in os.listdir(info_list) if f.endswith('.json')])
        #     self.single_file = False

        self.json_files = info_list
        self.root_dir = root_dir
        self.transform = transform

    # def __init__(self, info_list, root_dir, transform=None, single_file=None):
    #     self.json_files = sorted([
    #         os.path.join(info_list, f)
    #         for f in os.listdir(info_list)
    #         if f.endswith('.json')
    #     ])
    #     if single_file is not None:
    #         self.json_files = [f for f in self.json_files if os.path.basename(f) == single_file]
    #     self.image_root = root_dir
    #     self.transform = transform

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
            json_path = self.json_files[idx]
            with open(json_path, 'r') as f:
                data = json.load(f)

            print(json_path)

            # ===== Parse timestamp lists =====
            timestamps = [frame["Timestamp"] for frame in data["/Image"]]
            hr_entries = [
                (entry["Timestamp"], entry["Value"]["pulseRate"])
                for entry in data["/FullPackage"]
                if "pulseRate" in entry["Value"] and entry["Value"]["pulseRate"] > 0
            ]

            if len(hr_entries) == 0:
                raise ValueError(f"No valid pulseRate values found in {json_path}")

            frame_cnt = len(timestamps)
            framerate = 30.0
            total_clips = int((frame_cnt - 60 - 220) // 160) + 1

            video_x = np.zeros((total_clips, clip_frames, 128, 128, 3))
            clip_hr = np.zeros(total_clips, dtype=np.float32)

            for tt in range(total_clips):
                image_idx = tt * 160 + 60
                for i in range(clip_frames):
                    ts = timestamps[image_idx]
                    img_name = f"Image{ts}.png"
                    img_path = os.path.join(self.root_dir, img_name)

                    # tmp_image = cv2.imread(img_path)
                    # if tmp_image is None:
                    #     raise FileNotFoundError(f"Missing image: {img_path}")

                    if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                        print(f"[WARN] Missing or empty image: {img_path}")
                        tmp_image = np.zeros((128, 128, 3), dtype=np.uint8)
                        print(f"[INFO] Using black image instead.")
                    else:
                        tmp_image = cv2.imread(img_path)
                        if tmp_image is None:
                            print(f"[WARN] OpenCV failed to read image: {img_path}")
                            tmp_image = np.zeros((128, 128, 3), dtype=np.uint8)
                            print(f"[INFO] Using black image instead.")

                    tmp_image = cv2.resize(tmp_image, (132, 132), interpolation=cv2.INTER_CUBIC)[2:130, 2:130, :]
                    video_x[tt, i, :, :, :] = tmp_image
                    image_idx += 1

                # Find the closest HR value to the clip's starting frame timestamp
                clip_start_ts = timestamps[tt * 160 + 60]
                closest_hr = min(hr_entries, key=lambda x: abs(x[0] - clip_start_ts))[1]
                clip_hr[tt] = closest_hr

            sample = {
                'video_x': video_x,
                'framerate': framerate,
                'clip_average_HR_peaks': clip_hr
            }

            if self.transform:
                sample = self.transform(sample)

            return sample
