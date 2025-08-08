@echo off
REM Exit immediately on error
setlocal enabledelayedexpansion

REM Create conda environment
echo [1/7] Creating conda environment MVRBind...
call conda create -n MVRBind python=3.10 -y
if errorlevel 1 exit /b

REM Activate conda environment
echo [2/7] Activating environment MVRBind...
call conda activate MVRBind
if errorlevel 1 exit /b

REM Install PyTorch + CUDA 12.1
echo [3/7] Installing PyTorch with CUDA 12.1 compatibility...
call conda install pytorch=2.2.0 torchvision=0.17.0 torchaudio=2.2.0 pytorch-cuda=12.1 -c https://conda.anaconda.org/pytorch -c https://conda.anaconda.org/nvidia -y
if errorlevel 1 exit /b

REM Install PyTorch Geometric dependencies
echo [4/7] Installing PyTorch Geometric dependencies...
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric
if errorlevel 1 exit /b

REM Install general dependencies
echo [5/7] Installing other Python libraries...
pip install matplotlib
pip install biopython
pip install networkx
pip install scikit-learn
pip install pillow
pip install numpy==1.23.5
pip install pandas
if errorlevel 1 exit /b

REM Verification
echo [6/7] Verifying PyTorch and PyG installation...
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| PyG installed successfully')"

echo [7/7] Installation complete!
pause
