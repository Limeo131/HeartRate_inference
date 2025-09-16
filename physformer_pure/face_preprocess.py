# face_preprocess_pure.py
# ---- env: disable GPU/GL & reduce logs (must be set before importing mediapipe) ----
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
# -------------------------------------------------------------------

import re
import cv2
import json
import numpy as np
import mediapipe as mp

__all__ = ["preprocess_frame", "json2frames_preprocess"]

# ==================== Basic utilities ====================
def _img2landmark_478(img_rgb, face_mesh):
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

def _eye_center_and_angle(lm_xy):
    left  = lm_xy[_M478_PUPIL_LEFT].mean(axis=0)
    right = lm_xy[_M478_PUPIL_RIGHT].mean(axis=0)
    center = (left + right) / 2.0
    dy, dx = (right[1]-left[1]), (right[0]-left[0])
    angle_deg = np.degrees(np.arctan2(dy, dx))
    dist = float(np.linalg.norm(right - left))
    return center, angle_deg, dist

def _rotate(img_bgr, center_xy, angle_deg):
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D(tuple(center_xy), angle_deg, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h))

def _center_crop_128(frame_bgr):
    return cv2.resize(frame_bgr, (132, 132))[2:130, 2:130, :]

def _annotate(img_bgr, text):
    out = img_bgr.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return out

def _make_comparison(before_bgr, after_bgr, scale=2):
    if before_bgr.shape[:2] != (128,128):
        before_bgr = cv2.resize(before_bgr, (128,128))
    if after_bgr.shape[:2] != (128,128):
        after_bgr = cv2.resize(after_bgr, (128,128))
    comp = cv2.hconcat([_annotate(before_bgr,"Before"), _annotate(after_bgr,"After")])
    if scale != 1:
        comp = cv2.resize(comp, (comp.shape[1]*scale, comp.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    return comp

def _crop_square(img, center_xy, box_size, keep_size=True):
    h, w = img.shape[:2]
    cx, cy = center_xy
    half = box_size / 2.0
    left   = int(np.floor(cx - half))
    right  = int(np.ceil (cx + half))
    top    = int(np.floor(cy - half))
    bottom = int(np.ceil (cy + half))

    if keep_size:
        pad_left   = max(0, -left)
        pad_top    = max(0, -top)
        pad_right  = max(0, right  - w)
        pad_bottom = max(0, bottom - h)
        if pad_left or pad_top or pad_right or pad_bottom:
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE)
            left  += pad_left; right += pad_left
            top   += pad_top;  bottom+= pad_top
        return img[top:bottom, left:right]

    left   = max(0, left)
    right  = min(w, right)
    top    = max(0, top)
    bottom = min(h, bottom)
    side = min(right-left, bottom-top)
    right = left + side
    bottom= top + side
    return img[top:bottom, left:right]

# ==================== Face preprocessing ====================
def preprocess_frame(
    frame_bgr,
    face_mesh,
    out_size=128,
    crop_scale=4.2,   # Larger = "further"
    shift_y=0.30,     # Downward shift ratio (relative to eye distance)
    min_box=112       # Minimum box size for small faces
):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    lm = _img2landmark_478(img_rgb, face_mesh)
    if lm is None:
        return _center_crop_128(frame_bgr)

    h, w = frame_bgr.shape[:2]
    lm_xy = lm[:, :2] * [w, h]
    eye_center, angle, _ = _eye_center_and_angle(lm_xy)

    rot_bgr = _rotate(frame_bgr, eye_center, angle)

    rot_rgb = cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2RGB)
    lm2 = _img2landmark_478(rot_rgb, face_mesh)
    if lm2 is None:
        return _center_crop_128(rot_bgr)

    h2, w2 = rot_bgr.shape[:2]
    lm2_xy = lm2[:, :2] * [w2, h2]
    eye_center2, _, eye_dist2 = _eye_center_and_angle(lm2_xy)

    center_xy = eye_center2 + np.array([0.0, shift_y * eye_dist2], dtype=np.float32)
    box = float(np.clip(crop_scale * eye_dist2, min_box, min(h2, w2)))
    face_crop = _crop_square(rot_bgr, center_xy, box, keep_size=True)

    return cv2.resize(face_crop, (out_size, out_size))

# ==================== PURE: preprocessing based on JSON timestamps ====================
def _extract_timestamps(meta):
    """
    Compatible with two common structures:
    1) meta["/Image"] is list[{"Timestamp": ...}, ...]
    2) meta["Image"] same as above
    Returns ['ts1','ts2',...] (strings, keeping dataset naming convention)
    """
    for key in ("/Image", "Image"):
        if key in meta and isinstance(meta[key], list):
            lst = []
            for item in meta[key]:
                if isinstance(item, dict) and "Timestamp" in item:
                    ts = item["Timestamp"]
                    # Keep original form but ensure it is a string without spaces
                    lst.append(str(int(ts)) if isinstance(ts, (int,float)) and float(ts).is_integer() else str(ts))
            if lst:
                return lst
    return None

def _resolve_frame_path_by_ts(root_dir, ts_str, key=None):
    """
    Find original frame by Timestamp. Try several naming/dirs:
      Image{ts}.png  or  located in {root}/{key}/Image/
    """
    # Try both "123", "123.0"
    ts_cands = [ts_str]
    if re.fullmatch(r"\d+(\.0+)?", ts_str):
        ts_cands.append(str(int(float(ts_str))))

    names = []
    for t in ts_cands:
        names += [f"Image{t}.png", f"image{t}.png", f"{t}.png"]

    dirs = [
        root_dir,
        os.path.join(root_dir, "Image"),
        os.path.join(root_dir, "image"),
        os.path.join(root_dir, "images"),
    ]
    if key is not None:
        dirs += [
            os.path.join(root_dir, key),
            os.path.join(root_dir, key, "Image"),
            os.path.join(root_dir, key, "image"),
            os.path.join(root_dir, key, "images"),
        ]
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    return None

def json2frames_preprocess(
    json_path,
    orig_root,             # Original dataset root (contains Image{Timestamp}.png)
    out_root,              # Root for preprocessed outputs (save to out_root/Image{Timestamp}.png)
    use_face_preprocess=True,
    out_size=128,
    preprocess_kwargs=None,
    comparison_out_dir=None,
    save_comparison=False,
    comparison_pick="middle",   # "middle" / float(0~1) / int
    comparison_scale=2
):
    os.makedirs(out_root, exist_ok=True)

    with open(json_path, "r") as f:
        meta = json.load(f)

    timestamps = _extract_timestamps(meta)
    if not timestamps:
        print(f"[WARN] No timestamps found in {json_path}")
        return False

    key = os.path.splitext(os.path.basename(json_path))[0]

    face_mesh = None
    if use_face_preprocess:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
    if preprocess_kwargs is None:
        preprocess_kwargs = {}

    # Comparison image: sample 1 frame
    if save_comparison and comparison_out_dir is not None:
        os.makedirs(comparison_out_dir, exist_ok=True)
        if isinstance(comparison_pick, str) and comparison_pick == "middle":
            si = len(timestamps) // 2
        elif isinstance(comparison_pick, float) and 0.0 <= comparison_pick <= 1.0:
            si = int(len(timestamps) * comparison_pick)
        elif isinstance(comparison_pick, int):
            si = max(0, min(len(timestamps)-1, comparison_pick))
        else:
            si = len(timestamps) // 2

        ts = timestamps[si]
        src = _resolve_frame_path_by_ts(orig_root, ts, key=key)
        if src is not None:
            img = cv2.imread(src)
            before = _center_crop_128(img)
            after  = preprocess_frame(img, face_mesh, out_size=out_size, **preprocess_kwargs) if use_face_preprocess else before
            comp = _make_comparison(before, after, scale=comparison_scale)
            cv2.imwrite(os.path.join(comparison_out_dir, f"{key}_compare.png"), comp)

    # Batch process (output names must be consistent with Dataset: Image{Timestamp}.png)
    ok_cnt = 0
    for ts in timestamps:
        src = _resolve_frame_path_by_ts(orig_root, ts, key=key)
        if src is None:
            continue
        img = cv2.imread(src)
        if img is None:
            continue
        if use_face_preprocess:
            try:
                img = preprocess_frame(img, face_mesh, out_size=out_size, **preprocess_kwargs)
            except Exception:
                img = _center_crop_128(img)
        else:
            img = _center_crop_128(img)
        dst = os.path.join(out_root, f"Image{ts}.png")
        cv2.imwrite(dst, img)
        ok_cnt += 1

    if face_mesh is not None:
        face_mesh.close()

    print(f"[INFO] Preprocessed {ok_cnt}/{len(timestamps)} frames -> {out_root}")
    return ok_cnt > 0
