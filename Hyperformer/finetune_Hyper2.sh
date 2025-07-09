#!/bin/bash

python main_2.py \
    --batch-size 256 \
    --test-batch-size 128 \
    --num-worker 16 \
    --optimizer AdamW \
    --base-lr 0.0005 \
    --lr-decay-rate 0.3\
    --step 40 80 \
    --weight-decay 0.005 \
    --num-epoch  100\
    --weights hyperformer_pretrained_weights/ntu120/csub/joint/runs-134-131855.pt \
	--phase train \
	--save-score True \
	--config config/violence-dataset/joint_2.yaml \
	--model model.Hyperformer_2.Model \
	--work-dir work_dir/Hyperf_2 \
	--device=0
