#!/bin/bash

# ======= Config =======
WORKDIR=/home/siming/physformer_vipl
VIDEO_DIR=/mnt/vdb/vipl               # Video path (.avi)
CSV_INFO=/mnt/vdb/vipl/vipl_sample_info.csv
LOG_DIR=/home/siming/physformer_vipl/Inference_Physformer_VIPL   # Output results directory

CLIP_SIZE=160
CLIP_OVERLAP=60

# ======= Switch to working directory =======
cd $WORKDIR

# ======= Create output folder =======
mkdir -p $LOG_DIR

# ======= Start inference (background) =======
echo "Submitting VIPL PhysFormer job..."
nohup /root/anaconda3/bin/python run_physformer_vipl.py \
  --video_dir $VIDEO_DIR \
  --csv_info $CSV_INFO \
  --log $LOG_DIR \
  --clip_size $CLIP_SIZE \
  --clip_overlap $CLIP_OVERLAP \
  > $LOG_DIR/nohup.log 2>&1 &

echo "Submitted! You can check logs in $LOG_DIR/nohup.log"
