#!/bin/bash

python main_1.py \
    --batch-size 128 \
    --test-batch-size 128 \
    --num-worker 16 \
    --optimizer AdamW \
    --base-lr 0.01 \
    --lr-decay-rate 0.3\
    --step 40 80 120\
    --weight-decay 0.005\
    --num-epoch  150\
    --weights hyperformer_pretrained_weights/ntu120/csub/joint/runs-134-131855.pt\
	--phase train \
	--save-score True \
	--config config/violence-dataset/joint_1.yaml \
	--model model.Hyperformer_1.Model \
	--work-dir work_dir/Hyperf_1 \
	--device=0
