#!/bin/bash

# ======= 配置 =======
WORKDIR=/home/siming/physformer
VIDEO_DIR=/mnt/vdb/sample_video
CSV_INFO=/home/siming/physformer/sample_info.csv
LOG_DIR=/home/siming/physformer/Inference_Physformer_withpre

CLIP_SIZE=160
CLIP_OVERLAP=60

# ======= 切换目录 =======
cd $WORKDIR

# ======= 创建输出文件夹 =======
mkdir -p $LOG_DIR

# ======= 启动推理 (后台) =======
echo "Submitting PhysFormer job..."
nohup /root/anaconda3/bin/python run_physformer_withpre.py \
  --video_dir $VIDEO_DIR \
  --csv_info $CSV_INFO \
  --log $LOG_DIR \
  --clip_size $CLIP_SIZE \
  --clip_overlap $CLIP_OVERLAP \
  > $LOG_DIR/nohup.log 2>&1 &

echo "Submitted! You can check logs in $LOG_DIR/nohup.log"
