#!/usr/bin/env bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 8282 train_distributed.py --gpu_id '1' \
--gray_aug --gray_p 0.1 --scale_aug --scale_type 1 --scale_p 0.3 --epochs 1500 --lr_step 1200 --lr 3e-5 \
--batch_size 16 --num_patch 1 --threshold 0.35 --test_per_epoch 20 --num_queries 700 \
--dataset nwpu --crop_size 256 --pre None  --test_patch --save