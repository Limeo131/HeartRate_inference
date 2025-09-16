# ---- put these at the very top of face_preprocess.py ----
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1" # Disable MediaPipe GPU/GL pipeline
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silence TensorFlow WARNING
os.environ["GLOG_minloglevel"] = "2"      # Reduce glog output
os.environ["OPENCV_LOG_LEVEL"] = "SILENT" # Reduce OpenCV log
# ---------------------------------------------------------

import cv2
import numpy as np
import mediapipe as mp

__all__ = ["video2frames", "preprocess_frame"]

# ====== Internal tools ======
def _img2landmark_478(img_rgb, face_mesh):
    """Return (478,3) normalized coordinates (0~1) or None. img must be RGB."""
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    xs = np.array([p.x for p in lm], dtype=np.float32)
    ys = np.array([p.y for p in lm], dtype=np.float32)
    zs = np.array([p.z for p in lm], dtype=np.float32)
    return np.vstack([xs, ys, zs]).T  # (478,3)

_M478_PUPIL_LEFT  = [469, 470, 471, 472]
_M478_PUPIL_RIGHT = [476, 475, 474, 477]

def _eye_center_and_angle(lm_xy):  # lm_xy: (478,2) pixel coordinates
    left  = lm_xy[_M478_PUPIL_LEFT].mean(axis=0)
    right = lm_xy[_M478_PUPIL_RIGHT].mean(axis=0)
    center = (left + right) / 2.0
    dy, dx = (right[1] - left[1]), (right[0] - left[0])
    angle_deg = np.degrees(np.arctan2(dy, dx))
    dist = np.linalg.norm(right - left)
    return center, angle_deg, dist

def _rotate(img_bgr, center_xy, angle_deg):
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D(tuple(center_xy), angle_deg, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h))

def _crop_square(img, center_xy, box_size):
    """Crop a square centered at center_xy, auto adjust to avoid out-of-bounds."""
    h, w = img.shape[:2]
    cx, cy = center_xy
    half = box_size / 2.0
    left   = int(max(0, cx - half))
    right  = int(min(w, cx + half))
    top    = int(max(0, cy - half))
    bottom = int(min(h, cy + half))
    # After adjusting to edges, correct to square
    cur_w, cur_h = right - left, bottom - top
    side = min(cur_w, cur_h)
    right  = left + side
    bottom = top + side
    return img[top:bottom, left:right]

# ====== New: small utilities ======
def _center_crop_128(frame_bgr):
    """Same as your original logic: resize 132 then crop 128."""
    return cv2.resize(frame_bgr, (132, 132))[2:130, 2:130, :]

def _annotate(img_bgr, text):
    """Draw text at the top-left corner."""
    out = img_bgr.copy()
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return out

def _make_comparison(before_bgr, after_bgr, scale=2):
    """
    Make a horizontal comparison image: left=before, right=after.
    Default 2x upscaling for easier viewing.
    Input should be 128x128 BGR.
    """
    if before_bgr.shape[:2] != (128,128):
        before_bgr = cv2.resize(before_bgr, (128,128))
    if after_bgr.shape[:2] != (128,128):
        after_bgr = cv2.resize(after_bgr, (128,128))
    before = _annotate(before_bgr, "Before")
    after  = _annotate(after_bgr,  "After")
    comp = cv2.hconcat([before, after])
    if scale != 1:
        comp = cv2.resize(comp, (comp.shape[1]*scale, comp.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    return comp


# ====== Public functions ======
def preprocess_frame(
    frame_bgr,
    face_mesh,
    out_size=128,
    crop_scale=4.2,   # Originally 3.2 → enlarge crop box
    shift_y=0.35,     # Added: downward shift (proportional to eye distance)
    min_box=128       # Originally 96 → raise lower bound to avoid too small crops
):
    """
    Return the preprocessed 128x128 BGR image; fallback to center crop if failed.
    Steps: detect -> align rotation -> detect again -> crop based on eye distance
           (with optional downward shift/enlargement) -> resize to 128.
    """
    # 1) Initial detection
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    lm = _img2landmark_478(img_rgb, face_mesh)
    if lm is None:
        return cv2.resize(frame_bgr, (132, 132))[2:130, 2:130, :]

    h, w = frame_bgr.shape[:2]
    lm_xy = lm[:, :2] * [w, h]
    eye_center, angle, _ = _eye_center_and_angle(lm_xy)

    # 2) Alignment by rotation
    rot_bgr = _rotate(frame_bgr, eye_center, angle)

    # 3) Detection after alignment
    rot_rgb = cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2RGB)
    lm2 = _img2landmark_478(rot_rgb, face_mesh)
    if lm2 is None:
        return cv2.resize(rot_bgr, (132, 132))[2:130, 2:130, :]

    h2, w2 = rot_bgr.shape[:2]
    lm2_xy = lm2[:, :2] * [w2, h2]
    eye_center2, _, eye_dist2 = _eye_center_and_angle(lm2_xy)

    # 4) Enlarge + shift downward
    center_xy = eye_center2 + np.array([0.0, shift_y * eye_dist2], dtype=np.float32)
    box = float(np.clip(crop_scale * eye_dist2, min_box, min(h2, w2)))

    face_crop = _crop_square(rot_bgr, center_xy, box)

    # 5) Output standardized size
    return cv2.resize(face_crop, (out_size, out_size))

def video2frames(
    video_path, save_dir, target_frame_count,
    log_file=None, use_face_preprocess=True, out_size=128,
    comparison_out_dir=None,
    save_comparison=False,
    comparison_frame="middle",
    comparison_scale=2,
    preprocess_kwargs=None     # ★ New: pass crop parameters (dict)
):
    """
    Extract frames + (optional) face preprocessing. Output 128x128 PNG.
    Optional: also save one Before/After comparison image for this video
              (default middle frame).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0 or not cap.isOpened():
        msg = f"[SKIP] Cannot read frames: {video_path}"
        print(msg)
        if log_file: log_file.write(msg + '\n')
        return False

    os.makedirs(save_dir, exist_ok=True)

    face_mesh = None
    if use_face_preprocess:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    if preprocess_kwargs is None:
        preprocess_kwargs = {}

    # --- Comparison image ---
    if save_comparison and comparison_out_dir is not None:
        os.makedirs(comparison_out_dir, exist_ok=True)
        if isinstance(comparison_frame, str) and comparison_frame == "middle":
            sample_idx = max(0, total_frames // 2)
        elif isinstance(comparison_frame, float) and 0.0 <= comparison_frame <= 1.0:
            sample_idx = int(total_frames * comparison_frame)
        elif isinstance(comparison_frame, int):
            sample_idx = max(0, min(total_frames-1, comparison_frame))
        else:
            sample_idx = max(0, total_frames // 2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
        ret_s, sample = cap.read()
        if ret_s:
            before_img = _center_crop_128(sample)
            after_img  = (preprocess_frame(sample, face_mesh, out_size=out_size, **preprocess_kwargs)
                          if use_face_preprocess else _center_crop_128(sample))
            comp = _make_comparison(before_img, after_img, scale=comparison_scale)
            vid_key = os.path.splitext(os.path.basename(video_path))[0]
            comp_path = os.path.join(comparison_out_dir, f"{vid_key}_compare.png")
            cv2.imwrite(comp_path, comp)
            if log_file: log_file.write(f"[INFO] Saved comparison: {comp_path}\n")
        else:
            if log_file: log_file.write(f"[WARN] Could not read sample frame {sample_idx} for comparison.\n")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Normal frame extraction ---
    step = max(1e-6, total_frames / float(target_frame_count))
    for i in range(target_frame_count):
        frame_idx = round(i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            warn = f"[WARN] Could not read frame {frame_idx}"
            print(warn)
            if log_file: log_file.write(warn + "\n")
            continue

        if use_face_preprocess:
            try:
                frame = preprocess_frame(frame, face_mesh, out_size=out_size, **preprocess_kwargs)
            except Exception as e:
                if log_file: log_file.write(f"[WARN] preprocess failed at {frame_idx}: {e}\n")
                frame = _center_crop_128(frame)
        else:
            frame = _center_crop_128(frame)

        save_path = os.path.join(save_dir, f"image_{i+1:05d}.png")
        cv2.imwrite(save_path, frame)

    if face_mesh is not None:
        face_mesh.close()
    cap.release()

    msg = f"[INFO] Saved {target_frame_count} frames to {save_dir}"
    print(msg)
    if log_file: log_file.write(msg + '\n')
    return True
