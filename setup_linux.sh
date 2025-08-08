#!/bin/bash

set -e

conda create -n MVRBind python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MVRBind

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric

pip install matplotlib
pip install biopython
pip install networkx
pip install scikit-learn
pip install numpy==1.23.5
pip install pillow
pip install pandas


echo ">>> Verify Installation:"
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| PyG Successfully installed!')"
