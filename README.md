# 🌟 MVRBind

## **MVRBind: Multi-view Learning for RNA-Small Molecule Binding Site Prediction**

🔬 In this study, we introduce **MVRBind**, a **multi-view graph convolutional network** designed to predict RNA-small molecule binding sites. MVRBind constructs feature representations at three structural levels:

- 🧬 **Primary Structure**
- 🏗 **Secondary Structure**
- 📦 **Tertiary Structure**

✨ **Key Innovations:**
✅ A **multi-view fusion module** that learns distinct RNA structural features.
✅ **Cross-view feature fusion**, enabling integration of multi-scale representations.
✅ **Consistently outperforms baseline methods** across different experimental setups.
✅ **Accurate predictions** for RNA binding sites in both **holo and apo forms**.
✅ **Robust performance** across diverse RNA conformations.

These advancements position MVRBind as a powerful tool for **RNA-targeted drug discovery**, significantly enhancing the accuracy and reliability of binding site predictions. 🚀

![MVRBind Framework](mvrbind_framework.png)

---

## 📁 Project Structure

The project is organized as follows:

```
├── data_process/          # Scripts for data preprocessing
├── model_parameters/      # Trained model parameter files
├── pt/                    # Data files
├── environment.txt        # Dependency list (generated via pip)
├── environment.yml        # Dependency list (generated via Conda)
├── model.py               # Model definition script
├── predict.py             # Prediction script
├── train.py               # Model training script
├── docs/
│   ├── images/            # Directory for storing images
```

当然可以！你可以将两个标题都保留，一个作为总标题，一个作为子标题，结构清晰又专业。下面是更新后的版本，包含：

---

## ⚙️ Installation Guide

### 🧭 Step-by-Step Setup

---

### 🪟 For **Windows**

```bash
# 1. Clone the repository
git clone https://github.com/cschen-y/MVRBind
cd MVRBind

# 2. Install the environment (choose one method)

# ▶ Method A: Run the setup script
call setup_windows.bat

# ▶ Method B: Manual setup via environment.yml
conda env create -f environment.yml
conda activate mvrbind

# 3. Verify installation
python -c "import torch; import torch_geometric; print('✔ PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)"
```

---

### 🐧 For **Linux**

```bash
# 1. Clone the repository
git clone https://github.com/cschen-y/MVRBind
cd MVRBind

# 2. Install the environment (choose one method)

# ▶ Method A: Run the setup script
chmod +x setup_linux.sh
./setup_linux.sh

# ▶ Method B: Manual setup via environment.yml
conda env create -f environment.yml
conda activate mvrbind

# 3. Verify installation
python -c "import torch; import torch_geometric; print('✔ PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)"
```

---

## 🚀 Usage

### 🔥 Model Training

Train the model using the following command:

```bash
python train.py
```

📝 Trained model parameters will be saved in the `model_parameters/` directory.

### 🎯 Model Prediction

Use the trained model to make predictions:

```bash
python predict.py
```

