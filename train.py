import os
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import random
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from data_process.train_dataset import TrainDataset
from model import MVRBind
from predict import predict
from data_process.set_seed import set_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class EarlyStopping:
    def __init__(self, patience=200, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train():
    set_seed(1)
    dataset = TrainDataset(f'./pt/kop8')
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MVRBind(136)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = torch.nn.BCELoss()

    early_stopping = EarlyStopping(patience=200)

    train_losses = []
    val_losses = []
    best_mcc = 0

    for epoch in range(200):
        model.train()
        train_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        mcc_total = 0
        with torch.no_grad():
            for data in test_loader:
                output = model(data)
                loss = criterion(output, data.y)
                val_loss += loss.item()

                preds = (output > 0.5).cpu().numpy().astype(int)
                true = data.y.cpu().numpy()
                mcc = matthews_corrcoef(true, preds)
                mcc_total += mcc

        avg_val_loss = val_loss / len(test_loader)
        avg_mcc = mcc_total / len(test_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: LR = {current_lr:.6f}, Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, MCC = {avg_mcc:.4f}")

        if avg_mcc > best_mcc:
            best_mcc = avg_mcc
            torch.save(model.state_dict(), 'model_parameters/model.pt')

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()


    predict()

if __name__ == '__main__':
    train()
