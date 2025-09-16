import os
import cv2
import pandas as pd
import math
import shutil

def debug_video2frames(
    video_dir='/mnt/vdb/sample_video',
    csv_info='/home/siming/physformer/sample_info.csv',
    output_root='/home/siming/physformer/VIPL_frames',
    target_fps=30
):
    # ✅ 清空 VIPL_frames 文件夹
    if os.path.exists(output_root):
        print(f"[INFO] Clearing existing VIPL_frames at {output_root} ...")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
    print(f"[INFO] Created fresh output folder: {output_root}")

    # ✅ 读取 CSV & strip 列名
    df = pd.read_csv(csv_info)
    df.columns = df.columns.str.strip()
    print("[INFO] CSV columns:", df.columns.tolist())

    for idx, row in df.iterrows():
        video_name = str(row['file_name']).strip()

        # ✅ 自动补后缀
        if not video_name.endswith('.mp4'):
            video_name += '.mp4'

        gt_hr = float(row['hr_true'])


        # 检查 fps
        fps_val = row['fps']
        if pd.isna(fps_val) or math.isnan(fps_val):
            #print(f"[SKIP] Missing fps for: {video_name} -> Skipping this row")
            continue
        csv_fps = float(fps_val)

        # 检查 frame_cnt
        frame_cnt_val = row['frame_cnt']
        if pd.isna(frame_cnt_val) or math.isnan(frame_cnt_val):
            #print(f"[SKIP] Missing frame_cnt for: {video_name} -> Skipping this row")
            continue
        csv_frame_cnt = int(frame_cnt_val)

        # 检查文件是否存在
        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            #print(f"[SKIP] File not found: {video_path} -> Skipping this row")
            continue

        print(f"\n[INFO] Processing {video_name}")
        print(f" - CSV fps: {csv_fps}, CSV frame_cnt: {csv_frame_cnt}, gt_hr: {gt_hr}")

        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f" - Actual fps: {original_fps}, Actual frame_cnt: {total_frames}")

        if total_frames == 0:
            print(f"[SKIP] Cannot read frames for: {video_name}")
            continue

        if abs(total_frames - csv_frame_cnt) > 2:
            print(f" [WARNING] Frame count mismatch: CSV={csv_frame_cnt}, Actual={total_frames}")

        save_dir = os.path.join(output_root, video_name.replace('.mp4',''))
        os.makedirs(save_dir, exist_ok=True)
        print(f" [INFO] Output dir: {save_dir}")

        step = original_fps / target_fps

        frame_idx = 0
        save_idx = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                print(f" [INFO] Frame reading ended at index {frame_idx}")
                break

            # 中心裁剪
            frame = cv2.resize(frame, (132, 132))[2:130, 2:130, :]

            save_name = f"image_{save_idx:05d}.png"
            save_path = os.path.join(save_dir, save_name)

            ok = cv2.imwrite(save_path, frame)
            if not ok:
                print(f"[ERROR] Failed to save frame: {save_path}")
            else:
                print(f"[DEBUG] Saved: {save_path}")

            frame_idx += step
            save_idx += 1
            if int(frame_idx) >= total_frames:
                break

        cap.release()
        print(f" [DONE] Saved {save_idx} frames for {video_name} into {save_dir}")

if __name__ == '__main__':
    debug_video2frames(
        video_dir='/mnt/vdb/sample_video',
        csv_info='/home/siming/physformer/sample_info.csv',
        output_root='/home/siming/physformer/VIPL_frames',
        target_fps=30
    )
