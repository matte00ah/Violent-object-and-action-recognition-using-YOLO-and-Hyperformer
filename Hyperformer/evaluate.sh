#!/bin/bash
python main_3.py \
    --batch-size 128 \
    --test-batch-size 128 \
    --num-worker 16 \
    --optimizer AdamW \
    --base-lr 0.01 \
    --lr-decay-rate 0.3\
    --step 40 80 120\
    --weight-decay 0.005\
    --num-epoch  10\
    --weights /homes/mbulgarelli/Hyperformer/work_dir/Hyperf_3/runs-71-639.pt --phase test --save-score True --config /homes/mbulgarelli/Hyperformer/config/violence-dataset/joint_3.yaml --model model.Hyperformer_3.Model --work-dir /homes/mbulgarelli/Hyperformer/work_dir/Hyperf_3 --device=0
