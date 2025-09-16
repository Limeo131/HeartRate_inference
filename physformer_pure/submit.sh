#!/bin/bash

# ======= Configuration =======
WORKDIR=/home/siming/physformer_pure
VIDEO_DIR=/mnt/vdb/pure               # Path to video files (.avi)
CSV_INFO=/mnt/vdb/pure/pure_full_info.csv
LOG_DIR=/home/siming/physformer_pure/Inference_PURE_JSON   # Directory to save output results

CLIP_SIZE=160
CLIP_OVERLAP=60

# ======= Switch to working directory =======
cd $WORKDIR

# ======= Create output folder =======
mkdir -p $LOG_DIR

# ======= Launch inference (in background) =======
echo "Submitting VIPL PhysFormer job..."
nohup /root/anaconda3/bin/python run_physformer_pure_clip.py \
  --video_dir $VIDEO_DIR \
  --csv_info $CSV_INFO \
  --log $LOG_DIR \
  --clip_size $CLIP_SIZE \
  --clip_overlap $CLIP_OVERLAP \
  > $LOG_DIR/nohup.log 2>&1 &

echo "Submitted! You can check logs in $LOG_DIR/nohup.log"
