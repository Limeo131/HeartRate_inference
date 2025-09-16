import os
import json
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

clip_frames = 160 + 60  # 220 frames, including 60 frames overlap

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
        # if info_list ends with ".json", treat it as a single file
        # otherwise, read all json files in the directory
        # (currently disabled, using direct info_list input)

        self.json_files = info_list
        self.root_dir = root_dir
        self.transform = transform


    # Alternative initialization: directly load json files from directory
    # (kept here for reference, not active)
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
        # json_path = os.path.join(self.root_dir, self.json_files[idx])
        with open(json_path, 'r') as f:
            data = json.load(f)

        print(json_path)

        timestamps = [frame["Timestamp"] for frame in data["/Image"]]
        frame_cnt = len(timestamps)
        framerate = 30.0  # PURE dataset has fixed framerate
        
        pulseRates = [
            frame["Value"]["pulseRate"]
            for frame in data["/FullPackage"]
            if "pulseRate" in frame["Value"] and frame["Value"]["pulseRate"] > 0
        ]

        if len(pulseRates) == 0:
            raise ValueError(f"No valid pulseRate values found in {json_path}")

        clip_average_HR = float(np.mean(pulseRates))

        # Calculate number of clips, ensuring the last clip has at least clip_frames
        total_clips = int((frame_cnt - 60 - 220) // 160) + 1

        video_x = np.zeros((total_clips, clip_frames, 128, 128, 3))

        print(frame_cnt, total_clips)

        for tt in range(total_clips):
            image_idx = tt * 160 + 60
            for i in range(clip_frames):
                ts = timestamps[image_idx]
                img_name = f"Image{ts}.png"
                img_path = os.path.join(self.root_dir, img_name)

                tmp_image = cv2.imread(img_path)
                if tmp_image is None:
                    raise FileNotFoundError(f"Missing image: {img_path}")
                tmp_image = cv2.resize(tmp_image, (132, 132), interpolation=cv2.INTER_CUBIC)[2:130, 2:130, :]
                video_x[tt, i, :, :, :] = tmp_image
                image_idx += 1

        sample = {
            'video_x': video_x,
            'framerate': framerate,
            'clip_average_HR_peaks': clip_average_HR
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
