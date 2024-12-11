#!/bin/bash

#PJM -g ge58
#PJM -m e
#PJM -L rscgrp=share-debug
#PJM -L gpu=1

HOME=/work/ge58/e58004

eval "$(~/miniconda3/bin/conda shell.bash hook)"
module load cuda/12.1
module load cudnn/8.8.1

conda activate drug-discovery
cd ..
python ensemble_allexp3.py
conda deactivate