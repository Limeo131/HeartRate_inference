# from __future__ import print_function, division
# import os
# import torch
# import pandas as pd
# import cv2
# import numpy as np
# from torch.utils.data import Dataset
# import math

# class Normaliztion(object):
#     def __call__(self, sample):
#         video_x, framerate, clip_average_HR_peaks = (
#             sample['video_x'],
#             sample['framerate'],
#             sample['clip_average_HR_peaks']
#         )
#         new_video_x = (video_x - 127.5) / 128
#         return {
#             'video_x': new_video_x,
#             'framerate': framerate,
#             'clip_average_HR_peaks': clip_average_HR_peaks,
#             'file_name': sample['file_name']
#         }

# class ToTensor(object):
#     def __call__(self, sample):
#         video_x = sample['video_x']
#         framerate = sample['framerate']
#         clip_average_HR_peaks = sample['clip_average_HR_peaks']

#         video_x = video_x.transpose((0, 3, 1, 2))
#         video_x = np.array(video_x)

#         return {
#             'video_x': torch.from_numpy(video_x.astype(np.float32)).float(),
#             'framerate': torch.tensor([framerate]).double(),
#             'clip_average_HR_peaks': torch.tensor([clip_average_HR_peaks]).float(),
#             'file_name': sample['file_name']
#         }

# class VIPL(Dataset):
#     def __init__(self, info_list, root_dir, transform=None, clip_size=0, clip_overlap=0, single_file=None):
#         df = pd.read_csv(info_list)
#         df.columns = df.columns.str.strip()

#         if single_file:
#             if not single_file.endswith('.avi'):
#                 single_file += '.avi'
#             df = df[df['key'].str.strip().apply(lambda x: x if x.endswith('.avi') else x + '.avi') == single_file]
#             print(f"[INFO] Filtering: Only keeping {single_file}")

#         self.root_dir = root_dir
#         self.transform = transform
#         self.clip_size = clip_size
#         self.clip_overlap = clip_overlap

#         valid_clips = []

#         for idx, row in df.iterrows():
#             video_name = str(row['key']).strip()
#             if not video_name.endswith('.avi'):
#                 video_name += '.avi'
#             frame_dir = os.path.join(root_dir, video_name.replace('.avi', ''))

#             fps_val = row['fps']
#             frame_cnt_val = row['frame_cnt']
#             hr_val = row['hr_mean']

#             if pd.isna(fps_val) or math.isnan(fps_val):
#                 print(f"[SKIP] Missing fps for: {video_name}")
#                 continue
#             if pd.isna(frame_cnt_val) or math.isnan(frame_cnt_val):
#                 print(f"[SKIP] Missing frame_cnt for: {video_name}")
#                 continue
#             if not os.path.exists(frame_dir) or len(os.listdir(frame_dir)) == 0:
#                 print(f"[SKIP] Frame folder not found or empty: {frame_dir}")
#                 continue

#             frame_list = sorted(os.listdir(frame_dir))
#             total_frames = len(frame_list)

#             if clip_size > 0 and total_frames > clip_size:
#                 step = clip_size - clip_overlap
#                 for start in range(0, total_frames - clip_size + 1, step):
#                     valid_clips.append({
#                         'row': row,
#                         'start': start,
#                         'end': start + clip_size
#                     })
#             else:
#                 valid_clips.append({
#                     'row': row,
#                     'start': 0,
#                     'end': total_frames
#                 })

#         self.valid_clips = valid_clips
#         print(f"[INFO] Total valid samples (after clip split): {len(self.valid_clips)}")

#     def __len__(self):
#         return len(self.valid_clips)

#     def __getitem__(self, idx):
#         info = self.valid_clips[idx]
#         row = info['row']
#         start = info['start']
#         end = info['end']

#         video_name = str(row['key']).strip()
#         if not video_name.endswith('.avi'):
#             video_name += '.avi'
#         gt_hr = float(row['hr_mean'])

#         frame_dir = os.path.join(self.root_dir, video_name.replace('.avi',''))
#         frame_list = sorted(os.listdir(frame_dir))
#         selected_frames = frame_list[start:end]

#         frames = []
#         for f in selected_frames:
#             img = cv2.imread(os.path.join(frame_dir, f))
#             frames.append(img)

#         frames = np.stack(frames)  # [T,H,W,C]

#         sample = {
#             'video_x': frames,
#             'framerate': 30.0,
#             'clip_average_HR_peaks': gt_hr,
#             'file_name': video_name
#         }

#         if self.transform:
#             sample = self.transform(sample)
#         return sample


from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math


clip_frames = 160+60   


class Normaliztion (object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        video_x, framerate, clip_average_HR_peaks = sample['video_x'],sample['framerate'],sample['clip_average_HR_peaks']
        new_video_x = (video_x - 127.5)/128
        return {'video_x': new_video_x, 'framerate':framerate, 'clip_average_HR_peaks':clip_average_HR_peaks}




class ToTensor (object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        video_x, framerate, clip_average_HR_peaks = sample['video_x'],sample['framerate'],sample['clip_average_HR_peaks']

        # swap color axis because
        # numpy image: clip x depth x H x W x C
        # torch image: clip x C x depth X H X W
        video_x = video_x.transpose((0, 4, 1, 2, 3))
        video_x = np.array(video_x)
        
        framerate = np.array(framerate)
        
        clip_average_HR_peaks = np.array(clip_average_HR_peaks)
        
        return {'video_x': torch.from_numpy(video_x.astype(np.float32)).float(),'framerate': torch.from_numpy(framerate.astype(np.float64)).double(),'clip_average_HR_peaks': torch.from_numpy(clip_average_HR_peaks.astype(np.float32)).float()}

        


class VIPL (Dataset):

    def __init__(self, info_list, root_dir, transform=None, single_file=None):

        self.landmarks_frame = pd.read_csv(info_list) #, delimiter=' ', header=None)

        if single_file is not None:
            self.landmarks_frame = self.landmarks_frame[self.landmarks_frame['key'] == single_file].reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        #video_path = os.path.join(self.root_dir, str(self.landmarks_frame.iloc[idx, 0]))
        video_path = os.path.join(self.root_dir, str(self.landmarks_frame.loc[idx, 'key']))
        
        #total_clips = self.landmarks_frame.iloc[idx, 1]
        frame_cnt = self.landmarks_frame.loc[idx, 'frame_cnt']
#        total_clips = int((self.landmarks_frame.loc[idx, 'frame_cnt'] - 60) // 160)#+1
        total_clips = int((frame_cnt - 60 - 220) // 160) + 1
        
        video_x = self.get_single_video_x(video_path, total_clips,frame_cnt)
        
        #framerate  = self.landmarks_frame.iloc[idx, 2]
        framerate  = self.landmarks_frame.loc[idx, 'fps']
        
        #clip_average_HR  = self.landmarks_frame.iloc[idx, 3]
        clip_average_HR  = self.landmarks_frame.loc[idx, 'hr_mean']
		    
        
        sample = {'video_x': video_x, 'framerate':framerate, 'clip_average_HR_peaks':clip_average_HR}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_video_x(self, video_path, total_clips, frame_cnt):
        video_jpgs_path = video_path

        video_x = np.zeros((total_clips, clip_frames, 128, 128, 3))
        
        for tt in range(total_clips):
            image_id = tt*160 + 61

            image_count = min(clip_frames, frame_cnt - image_id)

            #print(frame_cnt, image_id, image_count)

            for i in range(image_count):
                s = "%05d" % image_id
                image_name = 'image_' + s + '.png'
    
                # face video 
                image_path = os.path.join(video_jpgs_path, image_name)
                
                tmp_image = cv2.imread(image_path)
                
                #if tmp_image is None:    # It seems some frames missing 
                #    tmp_image = cv2.imread(self.root_dir+'p30/v1/source2/image_00737.png')
                    
                tmp_image = cv2.resize(tmp_image, (132, 132), interpolation=cv2.INTER_CUBIC)[2:130, 2:130, :]
                
                video_x[tt, i, :, :, :] = tmp_image  
                            
                image_id += 1
   
        return video_x