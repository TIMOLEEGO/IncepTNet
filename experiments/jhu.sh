#!/usr/bin/env bash
torchrun --nproc_per_node=2 --master_port 3218 train_distributed.py --gpu_id '1,2' \
--gray_aug --gray_p 0.3 --scale_aug --scale_type 1 --scale_p 0.3 --epochs 1200 --lr_step 1200 --lr 3e-5  \
--batch_size 16 --num_patch 1 --threshold 0.35 --test_per_epoch 20 \
--dataset jhu --crop_size 256 --pre None --test_patch --save