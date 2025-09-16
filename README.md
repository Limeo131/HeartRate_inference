# HeartRate_inference

- Physformer: need to download trained checkpoint 'Physformer_VIPL_fold1.pkl' (https://drive.google.com/file/d/1jBSbM88fA-beaoVi8ILFyL0SvVVMA9c9/view)

- Physformer_vipl: Inference for VIPL run_physformer_vipl_withpre.py

- Physformer_pure: Inference for PURE run_physformer_pure_clip_withpre.py

- Physformer: Inference for Wechat mini_APP run_physformer_withpre.py



# HeartRate_inference

Remote heart rate (HR) inference from face video with **PhysFormer**. This repo provides ready‑to‑run inference pipelines for two popular datasets (VIPL‑HR and PURE), plus a lightweight script for a WeChat mini‑app scenario.

> If you use the code or ideas here, please also cite the PhysFormer paper (CVPR 2022) listed at the end.

---

## ✨ Features

- **PhysFormer-based inference** for multiple sources:
  - `physformer_vipl/`: inference on **VIPL‑HR** videos.
  - `physformer_pure/`: inference on **PURE** frames + JSON metadata.
  - `physformer/`: a simple entry for a WeChat mini‑app style demo.
- **Preprocessing** (see `preprocess.py`): face detection/cropping, spatial normalization, clip segmentation.
- **Outputs**:
  - rPPG waveform per clip (e.g., `.mat` files for downstream analysis).
  - Per‑clip HR (CSV) for quick evaluation/plotting.
- **Pluggable checkpoints**: load a pre‑trained PhysFormer weight file and run.

---

## 📁 Repository structure

```
HeartRate_inference/
├── physformer/                 # Inference entry for WeChat mini-app (script: run_physformer_withpre.py)
├── physformer_pure/            # Inference on PURE dataset (script: run_physformer_pure_clip_withpre.py)
├── physformer_vipl/            # Inference on VIPL-HR dataset (script: run_physformer_vipl_withpre.py)
├── preprocess.py               # Shared preprocessing utilities
└── README.md
```

> The exact CLI flags are defined inside each `run_*.py` script. Use `-h/--help` on any script to see all options.

---

## 🚀 Quick start

### 0) Environment

- Python ≥ 3.8
- PyTorch (CUDA optional but recommended)
- Common Python libs: `numpy`, `scipy`, `pandas`, `matplotlib`, `opencv-python`, `tqdm`, `scikit-image`, `imageio`

```bash
# create a clean env (example with conda)
conda create -n hr_infer python=3.10 -y
conda activate hr_infer

# install PyTorch (choose the right CUDA version from pytorch.org)
pip install torch torchvision torchaudio  # add --index-url if using CUDA wheels

# install general deps
pip install numpy scipy pandas matplotlib opencv-python tqdm scikit-image imageio
```

> If you already have a preferred environment, install only the missing packages. If a `requirements.txt` is later added to the repo, prefer `pip install -r requirements.txt`.

### 1) Download checkpoint

Download the PhysFormer checkpoint (example: `Physformer_VIPL_fold1.pkl`) from Google Drive:

- https://drive.google.com/file/d/1jBSbM88fA-beaoVi8ILFyL0SvVVMA9c9/view and place it anywhere, then pass its path via `--ckpt` (or the equivalent argument used by the script).

### 2) Prepare data

**VIPL‑HR**

- Videos are `.avi` files.
- A metadata CSV is expected, typically named something like `vipl_sample_info.csv` with columns such as:
  - `key` – video name (without extension)
  - `frame_cnt` – total frames
  - `fps` – native frames per second
  - `hr_mean` – reference/ground‑truth HR (if available)

Organize as:
```
/path/to/vipl
  ├── videos/                  # *.avi
  └── vipl_sample_info.csv
```

**PURE**

- PURE stores **frames as images** and **one JSON per video**.
- The JSON contains timestamped indices for mapping frames to clips and may include ground‑truth HR traces. Organize as:
```
/path/to/pure
  ├── frames/                  # Image{timestamp}.png (all frames for all videos)
  └── meta/                    # e.g., 01-01.json, 02-02.json, ...
```
- The dataloader maps each video to its frames using timestamps listed in the JSON.

> TIP: Make sure FPS and timestamp fields are correct—wrong FPS leads to wrong BPM.

### 3) Run inference

> Use `python <script> -h` to check the exact argument names. Below are typical examples.

**VIPL‑HR**

```bash
python physformer_vipl/run_physformer_vipl_withpre.py   --video_root /path/to/vipl/videos   --csv /path/to/vipl/vipl_sample_info.csv   --ckpt /path/to/checkpoints/Physformer_VIPL_fold1.pkl   --out outputs/vipl   --save_wave 1 --save_hr_csv 1
```

**PURE**

```bash
python physformer_pure/run_physformer_pure_clip_withpre.py   --frames_root /path/to/pure/frames   --json_root /path/to/pure/meta   --ckpt /path/to/checkpoints/Physformer_VIPL_fold1.pkl   --out outputs/pure   --save_wave 1 --save_hr_csv 1
```

**WeChat mini‑app demo**

```bash
python physformer/run_physformer_withpre.py   --video /path/to/video.mp4   --ckpt /path/to/checkpoints/Physformer_VIPL_fold1.pkl   --out outputs/demo
```

### 4) Outputs

- `outputs/*/clip_hr.csv` – per‑clip HR with columns like: `key, clip_idx, t_start, t_end, hr_pred_bpm`.
- `outputs/*/waves/<key>_clip<k>.mat` – rPPG waveform for each processed clip.
- Optional static plots (if enabled in the script).

---

## 🧩 Preprocessing details (high level)

- **Face region** detection & robust cropping
- **Spatial normalization** (resize to a fixed input size expected by PhysFormer)
- **Clip segmentation** (fixed length with stride; uses the video’s native FPS or metadata)
- **Standardization** for network input

You can customize these steps in `preprocess.py` or inside each `run_*.py` script.

---

## ⚠️ Notes & tips

- **FPS matters**: ensure the FPS in your CSV/metadata matches the actual videos.
- **Quality gates**: heavy motion, occlusion, tiny face regions, or flicker can degrade HR; filtering poor segments before inference improves stability.
- **Reproducibility**: set seeds and fixed clip boundaries when you want strict comparability across runs.
- **Device**: run on GPU when available (e.g., `CUDA_VISIBLE_DEVICES=0`).

---

## 📚 Citation

Please consider citing **PhysFormer** if this repo helps your research:

```
@inproceedings{yu2022physformer,
  title     = {PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer},
  author    = {Yu, Zitong and Shen, Yuming and Shi, Jingang and Zhao, Hengshuang and Torr, Philip and Zhao, Guoying},
  booktitle = {CVPR},
  year      = {2022}
}
```

---

## 📜 License

This repository currently does not include an explicit license file. If you plan to release it publicly, consider adding a suitable open‑source license (e.g., MIT/Apache‑2.0) or clarifying usage permissions.

---

## 🙏 Acknowledgements

- PhysFormer authors for the original model and training recipe.
- VIPL‑HR and PURE dataset maintainers and contributors.

---

## 🗺️ Roadmap ideas (optional)

- Add a `requirements.txt` / `environment.yml` with pinned versions.
- Provide sample config + CLI flags in each subfolder README.
- Add unit tests for preprocessing.
- Dockerfile for reproducible runs.
- Per‑clip evaluation scripts (MAE/RMSE/CI) and plotting utilities.
