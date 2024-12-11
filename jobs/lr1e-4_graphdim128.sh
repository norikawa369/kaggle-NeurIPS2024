#!/bin/bash

#PJM -g ge58
#PJM -m e
#PJM -L rscgrp=regular-a
#PJM -L node=1

HOME=/work/ge58/e58004

eval "$(~/miniconda3/bin/conda shell.bash hook)"
module load cuda/12.1
module load cudnn/8.8.1

conda activate drug-discovery
cd ../
python main.py  --save_dir /work/ge58/e58004/kaggle-comps/kaggle-leashbio-belka/exp/exp3/output/lr1e-4_graphdim128 --epochs 30 --batch_size 5000 --iters_accumulate 1 --num_workers 72 --dropout 0.1 --init_lr 1e-4 --node_dim 80 --edge_dim 16 --graph_dims 128
conda deactivate   