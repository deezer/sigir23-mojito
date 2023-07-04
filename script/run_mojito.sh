#!/bin/bash

# Preprocess
python data_misc/kcore_interactions.py

# Training
mkdir -p exp/logs/ml1m/10ucore_5icore/seqlen50/samples_step100
echo "python -m mojito train --verbose -p configs/ml1m.json"
CUDA_VISIBLE_DEVICES=0 python -m mojito train --verbose -p configs/ml1m.json \
>& exp/logs/ml1m/10ucore_5icore/seqlen50/samples_step100/mojito_lr0.001_batch512_epoch100_dim64_l2emb0.0_nblocks2_nheads2_drop0.3_trans0.5_glob0.1.log

# Evaluation (the best epoch is logged at the end of training step)
CUDA_VISIBLE_DEVICES=0 python -m mojito eval --verbose -p configs/ml1m.json --best_epoch 99

######################

# Training
echo "python -m mojito train --verbose -p configs/amzb.json"
mkdir -p exp/logs/amz_book/30ucore_20icore/seqlen50/samples_step5
CUDA_VISIBLE_DEVICES=0 python -m mojito train --verbose -p configs/amzb.json \
>& exp/logs/amz_book/30ucore_20icore/seqlen50/samples_step5/mojito_lr0.001_batch512_epoch100_dim64_l2emb0.0_nblocks2_nheads2_drop0.5_trans0.5_glob0.1_l2u0_l2i0.log

# Evaluation (the best epoch is logged at the end of training step)
CUDA_VISIBLE_DEVICES=0 python -m mojito eval --verbose -p configs/amzb.json --best_epoch 97

######################

# Training
echo "python -m mojito train --verbose -p configs/lfm1b.json"
mkdir -p exp/logs/lfm1b/300ucore_500icore/seqlen50/samples_step300
CUDA_VISIBLE_DEVICES=0 python -m mojito train --verbose -p configs/lfm1b.json \
>& exp/logs/lfm1b/300ucore_500icore/seqlen50/samples_step300/mojito_lr0.001_batch512_epoch100_dim64_l2emb0.0_nblocks2_nheads2_drop0.3_trans0.5_glob0.1_l2u0_l2i0.log

# Evaluation (the best epoch is logged at the end of training step)
CUDA_VISIBLE_DEVICES=0 python -m mojito eval --verbose -p configs/lfm1b.json --best_epoch 90
