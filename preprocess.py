import os
import re
import time
import numpy as np
import pandas as pd
import cv2
# from PIL import Image
# import pillow_heif
import matplotlib.pyplot as plt
import tqdm

import mediapipe as mp




def img2landmark_478(img, face_mesh=None):
    # landmarks: (478, 3), float, 0-1
    h, w, _ = img.shape
    face_mesh_status = True
    if face_mesh is None:
        face_mesh_status = False
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    try:
        landmarks = face_mesh.process(img).multi_face_landmarks[0].landmark
        xs = np.array([i.x for i in landmarks])
        ys = np.array([i.y for i in landmarks])
        zs = np.array([i.z for i in landmarks])
        landmarks = np.vstack([xs, ys, zs]).T
        if not face_mesh_status:
            face_mesh.close()
        return landmarks
    except Exception as err:
        return None

# def vid2landmark_478(vid):
#     face_mesh = mp.solutions.face_mesh.FaceMesh(
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )
#     res = {}
#     frame_cnt, h, w, _ = vid.shape
#     for i in tqdm.tqdm(range(frame_cnt)):
#         d = img2landmark_478(vid[i], face_mesh)
#         # res.append(d.reshape(-1, ))
#         res[i] = d.reshape(-1, )
#     cols = ['{1}{0}'.format(i, j) for i in range(478) for j in list('xyz')]
#     df = pd.DataFrame(columns=cols)
#     for i in range(frame_cnt):
#         df.loc[i] = res[i]
#     df = df.reset_index().rename(columns={'index': 'ind'})
#     return df

def get_face_rotate_status(landmarks):
    landmarks = landmarks[:, : 2]
    left_eye = landmarks[m478_region['pupil_left']].mean(axis=0)
    right_eye = landmarks[m478_region['pupil_right']].mean(axis=0)
    eye_center = (left_eye + right_eye) / 2

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    res = {'center': eye_center, 'angle': angle}
    return res

def rotate_img(img, center, angle):
    h, w, _ = img.shape
    if (center[0] <= 1) and (center[1] <= 1):
        center = center * [w, h]
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1)
    img_rotate = cv2.warpAffine(img, M, (w, h))
    return img_rotate

def crop_face(img, landmarks, target_w, target_h):
    h, w, _ = img.shape
    landmarks = landmarks[:, : 2]
    face_center = landmarks[4] # nose tip
    if (face_center[0] <= 1) and (face_center[1] <= 1):
        face_center = face_center * [w, h]
    
    left = face_center[0] - target_w / 2
    right = face_center[0] + target_w / 2
    top = face_center[1] - target_h / 2
    bottom = face_center[1] + target_h / 2
    return img[int(top): int(bottom), int(left): int(right)]


# def crop_face(img, landmarks):
#     def f(lower_0, upper_0, lower_1, upper_1):
#         l1 = upper_1 - lower_1
#         if lower_1 < lower_0:
#             return lower_0, lower_0 + l1
#         elif upper_1 > upper_0:
#             return upper_0 - l1, upper_0
#         else:
#             return lower_1, upper_1
#     h, w, _ = img.shape
#     landmarks = landmarks[:, : 2]
#     if landmarks[:, 0].max() <= 1:
#         landmarks = landmarks * [w, h]
#     l, t = landmarks.min(axis=0)
#     r, b = landmarks.max(axis=0)
#     # l = landmarks[:, 0].min()
#     # r = landmarks[:, 0].max()
#     # t = landmarks[:, 1].min()
#     # b = landmarks[:, 1].max()
#     face_w = r - l
#     face_h = b - t
#     t2 = max(0, t - face_h * 0.2)
#     b2 = min(h, b + face_h * 0.1)
#     face_h2 = b2 - t2
#     if face_h2 > w:
#         face_h2 = w
#         gap = (face_h2 - face_h) / 3
#         t2, b2 = f(0, h, t - gap * 2, b + gap)
#         l2 = 0
#         r2 = w
#     else:
#         gap = (face_h2 - face_w) / 2
#         l2, r2 = f(0, w, l - gap, r + gap)
#         gap = (face_h2 - face_h) / 3
#         t2, b2 = f(0, h, t - gap * 2, b + gap)
#     boundary1 = [l, r, t, b]
#     boundary2 = [l2, r2, t2, b2]
#     img_crop = img[int(t2): int(b2), int(l2): int(r2)]
#     return img_crop, boundary1, boundary2

def preprocess(img):
    landmarks = img2landmark_478(img)
    if landmarks is None:
        return False, None, None, None
    landmarks = landmarks[:, : 2] * [img.shape[1], img.shape[0]]
    res = get_face_rotate_status(landmarks)
    # print (res)
    center = res['center']
    angle = res['angle']
    img_rotate = rotate_img(img, center, angle)
    # cv2.imwrite('img_rotate.png', img_rotate[:, :, : : -1])

    landmarks = img2landmark_478(img_rotate)
    img_crop, b1, b2 = crop_face(img_rotate, landmarks)
    # print (img_crop.shape)

    landmarks = img2landmark_478(img_crop)
    return True, img_crop, landmarks, res


# def plot(img, landmarks, roi_m, roi_type='region', save_path=None):
#     '''
#     roi_m: {key: inds}
#     '''
#     landmarks = landmarks.copy()
#     l = landmarks.shape[1]
#     if l == 3:
#         landmarks = landmarks[:, : 2]

#     if landmarks.max() <= 2: # facemesh landmark
#         h, w = img.shape[: 2]
#         landmarks *= [w, h]
#     plt.figure(figsize=(10, 10))
#     plt.imshow(img)
#     for k, v in roi_m.items():
#         temp = landmarks[v][:, : 2].copy()
#         if roi_type in ['region']:
#             plt.plot(np.append(temp[:, 0], temp[0, 0]), np.append(temp[:, 1], temp[0, 1]), label=k)
#         else:
#             plt.plot(temp[:, 0], temp[:, 1], label=k)
#     plt.legend()
#     if save_path:
#         plt.savefig(save_path)
#         plt.close()

m68 = {
    'eye_left': [36, 37, 38, 39, 40, 41], 
    'eye_right': [42, 43, 44, 45, 46, 47], 
    'lip_upper': [48, 49, 50, 51, 52, 53, 54, 64, 63, 62, 61, 60], 
    'lip_lower': [48, 60, 67, 66, 65, 64, 54, 55, 56, 57, 58, 59], 
}

m478_region = {
    'forehead_left': [109, 107, 66, 105, 63, 70, 21, 54, 103, 67], 
    'forehead_center': [109, 10, 338, 336, 285, 8, 55, 107], 
    'forehead_right': [338, 336, 296, 334, 293, 300, 251, 284, 332, 297], 

    'eye_left': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7], 
    'eye_right': [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249], 
    'eye_left_outer': [124, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 35], 
    'eye_right_outer': [353, 445, 444, 443, 442, 441, 413, 464, 453, 452, 451, 450, 449, 448, 265], 

    'pupil_left': [469, 470, 471, 472], 
    'pupil_right': [476, 475, 474, 477], 

    'cheek_left_top': [234, 116, 117, 118, 119, 120, 47, 209, 203, 206, 207, 132, 93], 
    'cheek_left_bottom': [132, 207, 206, 216, 212, 202, 169, 136, 172, 58], 
    'cheek_right_top': [454, 345, 346, 347, 348, 349, 277, 429, 423, 426, 427, 361, 323], 
    'cheek_right_bottom': [361, 427, 426, 436, 432, 422, 394, 365, 397, 288], 

    'lip_upper': [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 
        308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78
    ], 
    'lip_lower': [
        61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 
        375, 321, 405, 314, 17, 84, 181, 91, 146
    ], 

    'nose': [168, 417, 412, 399, 429, 279, 294, 327, 2, 98, 64, 49, 209, 174, 188, 193], 

    'mouth': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146], 
    'mouth_inner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95], 
    'mouth_outer': [57, 186, 92, 165, 167, 164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43], 

    'chin': [202, 204, 194, 201, 200, 421, 418, 424, 422, 394, 379, 378, 400, 377, 152, 148, 176, 149, 150, 169], 

    'oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109], 
}


m478_line = {
    'forehead_1': [54, 103, 67, 109, 10, 338, 297, 332, 284], 
    'forehead_2': [68, 104, 69, 108, 151, 337, 299, 333, 298], 
    
    'brow_left_upper': [70, 63, 105, 66, 107], 
    'brow_left_lower': [46, 53, 52, 65, 55], 
    'brow_right_upper': [300, 293, 334, 296, 336], 
    'brow_right_lower': [276, 283, 282, 295, 285], 
    
    'eyelid_left_upper': [33, 246, 161, 160, 159, 158, 157, 173, 133], 
    'eyelid_left_lower': [33, 7, 163, 144, 145, 153, 154, 155, 133], 

    'eyelid_right_upper': [263, 466, 388, 387, 386, 385, 384, 398, 362], 
    'eyelid_right_lower': [263, 249, 390, 373, 374, 380, 381, 382, 362], 

    'pupil': [468, 473], 

    'cheek_left_1': [116, 111, 117, 118, 119, 120, 121], 
    'cheek_left_2': [123, 50, 101, 100, 47], 
    'cheek_left_3': [147, 187, 205, 36, 142, 126], 
    
    'cheek_right_1': [345, 340, 346, 347, 348, 349, 350], 
    'cheek_right_2': [352, 280, 330, 329, 277], 
    'cheek_right_3': [376, 411, 425, 266, 371, 355], 
    
    'mouth_corner_left': [186, 57], 
    'mouth_corner_right': [410, 287], 

    'chin_1': [212, 202, 204, 194, 201, 200, 421, 418, 424, 422, 432], 
    'chin_2': [214, 210, 211, 32, 208, 199, 428, 262, 431, 430, 434], 
    'chin_3': [138, 135, 169, 170, 140, 171, 175, 396, 369, 395, 394, 364, 367], 

    'axis_v': [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152], 
    'axis_h': [127, 356], 
}

m478_point = {
    'forehead_top': 10, 
    'glabella': 9, 
    'temple_left': 127, 
    'temple_right': 356, 
    
    'eye_left_outer': 33, # 130, 33
    'eye_left_inner': 133, # 243, 133
    'eye_left_top': 159, 
    'eye_left_bottom': 145, 
    'pupil_left': 468, 
    
    'eye_right_outer': 263, # 359, 263
    'eye_right_inner': 362, # 463, 362
    'eye_right_top': 386, 
    'eye_right_bottom': 374, 
    'pupil_right': 473, 
    
    'nasion': 6, 
    'nose_tip': 4, 
    'subnasale': 2, 
    'nostril_left': 64, # 48, 64
    'nostril_right': 294, # 278, 294
    
    'cheek_left': 123, 
    'cheek_right': 352, 
    
    'mouth_left': 61, 
    'mouth_right': 291, 
    'lip_upper': 0, 
    'stomion': 13, 
    'lip_lower': 17, 
    'chin': 152, 
    'jaw_left': 172, 
    'jaw_right': 397, 
}



if __name__ == '__main__':
    pass
    # get landmarks
    # landmarks = img2landmark_478(img)
    # try:
    #     print (landmarks.shape)
    #     print ('478-landmark detection success')
    #     np.save('{0}_landmark_478.npy'.format(key), landmarks)
    #     # plot(img, landmarks, m478_line, 'line', '{0}_landmark_478.png'.format(key))
    # except:
    #     print ('no face')



    # # get rotation angle
    # path = 'user5_normal.png'
    # img = cv2.imread(path)[:, :, : : -1]
    # landmarks = img2landmark_478(img)
    # landmarks = landmarks[:, : 2] * [img.shape[1], img.shape[0]]
    # res = get_face_rotate_status(landmarks)
    # print (res)
    # center = res['center']
    # angle = res['angle']
    # img_rotate = rotate_img(img, center, angle)
    # cv2.imwrite('img_rotate.png', img_rotate[:, :, : : -1])

    # # crop img
    # path = 'user2_normal.png'
    # img = cv2.imread(path)[:, :, : : -1]
    # landmarks = img2landmark_478(img)
    # img_crop, b1, b2 = crop_face(img, landmarks)
    # print (img.shape, img_crop.shape, b1, b2)
    # cv2.imwrite('img_crop.png', img_crop[:, :, : : -1])











