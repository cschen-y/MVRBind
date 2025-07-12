@echo off
REM 遇到错误立即退出（批处理没有 set -e，手动模拟）
setlocal enabledelayedexpansion

REM 创建 conda 环境
echo [1/7] 创建 conda 环境 MVRBind...
call conda create -n MVRBind python=3.10 -y
if errorlevel 1 exit /b

REM 激活 conda 环境
echo [2/7] 激活环境 MVRBind...
call conda activate MVRBind
if errorlevel 1 exit /b

REM 安装 PyTorch + CUDA 12.1
echo [3/7] 安装 PyTorch 与 CUDA 12.1 兼容版本...
call conda install pytorch=2.2.0 torchvision=0.17.0 torchaudio=2.2.0 pytorch-cuda=12.1 -c https://conda.anaconda.org/pytorch -c https://conda.anaconda.org/nvidia -y
if errorlevel 1 exit /b

REM 安装 PyTorch Geometric 依赖
echo [4/7] 安装 PyTorch Geometric 依赖项...
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric
if errorlevel 1 exit /b

REM 安装常规依赖
echo [5/7] 安装其他 Python 库...
pip install matplotlib
pip install biopython -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install networkx -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.23.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 exit /b

REM 验证
echo [6/7] 验证 PyTorch 和 PyG 安装...
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| PyG 安装成功')"

echo [7/7] 安装完成！
pause
