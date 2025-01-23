# MVRBind

MVRBind: Multi-view Learning for RNA-Small Molecule Binding Site Prediction.

---

## Project Structure

The project structure is as follows:

```plaintext
├── data_process/          # Scripts for data preprocessing
├── model_parameters/      # Trained model parameter files
├── pt/                    # Data files
├── environment.txt        # Dependency list (generated via pip)
├── environment.yml        # Dependency list (generated via Conda)
├── model.py               # Model definition script
├── predict.py             # Prediction script
├── train.py               # Model training script
```

---

## Installation Guide

### Installation with Conda
1. Clone this repository:
   ```bash
   git clone https://github.com/cschen-y/MVRBind
   cd MVRBind
   ```

2. Create a Conda environment and install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate mvrbind
   ```

### Installation with pip
1. Clone this repository:
   ```bash
   git clone https://github.com/cschen-y/MVRBind
   cd MVRBind
   ```

2. Install dependencies:
   ```bash
   pip install -r environment.txt
   ```

---

## Usage

### Model Training
Train the model using the following command:
```bash
python train.py
```
Trained model parameters will be saved in the `model_parameters/` directory.

### Model Prediction
Use the trained model to make predictions:
```bash
python predict.py
```
---
