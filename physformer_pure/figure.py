import os
import json
import cv2

def extract_10th_frame(json_path, image_dir, save_dir):
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    timestamps = [str(entry['Timestamp']) for entry in data.get('/Image', [])]

    if len(timestamps) < 10:
        print(f"[SKIP] Less than 10 frames in {json_path}")
        return

    ts = timestamps[9]  # 10th frame (index 9)
    image_name = f"Image{ts}.png"
    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        print(f"[SKIP] Missing image: {image_name}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] Cannot read image: {image_name}")
        return

    img = cv2.resize(img, (132, 132))[2:130, 2:130, :]  # center crop 128x128
    os.makedirs(save_dir, exist_ok=True)

    save_name = os.path.basename(json_path).replace(".json", "_frame10.png")
    save_path = os.path.join(save_dir, save_name)
    cv2.imwrite(save_path, img)

    print(f"[INFO] Saved: {save_path}")

if __name__ == "__main__":
    json_root = "/mnt/vdb/pure"  # Directory containing JSON and images
    save_root = "/home/siming/physformer_pure/Inference_PURE_JSON/frame10s"

    for file in os.listdir(json_root):
        if file.endswith(".json"):
            json_path = os.path.join(json_root, file)
            extract_10th_frame(json_path, json_root, save_root)
