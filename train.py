import os
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from data_process.train_dataset import TrainDataset
from model import MVRBind
import random
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from predict import predict
from data_process.set_seed import set_seed


def train():
    dataset = TrainDataset(f'./pt/kop8')
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_dataset, batch_size=28, shuffle=True)
    test_t_loader = DataLoader(test_dataset, batch_size=18, shuffle=True)
    model = MVRBind(136)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00006)
    crit = torch.nn.BCELoss()
    test_t_mcc = 0
    for epoch in range(80):
        # Initialize variables to track total loss and MCC
        loss_all = 0
        all_mcc = 0

        # Training loop for each batch in the training set
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            label = data.y
            loss = crit(output, label)
            loss.backward()
            loss_all += loss.item()
            optimizer.step()

        # Calculate average loss for the current epoch
        avg_loss = loss_all / len(train_loader)

        # Validation loop for each batch in the test set
        for data in test_t_loader:
            output = model(data)
            threshold = 0.5
            binary_predictions = (output.detach().numpy() > threshold).astype(int)
            mcc_value = matthews_corrcoef(data.y, binary_predictions)
            all_mcc += mcc_value

        # Calculate average MCC for the current epoch
        avg_mcc = all_mcc / len(test_t_loader)

        # Print average loss and MCC for the current epoch
        print(f"Epoch {epoch + 1}/80: Loss = {avg_loss:.4f}, MCC = {avg_mcc:.4f}")

        # Save the model if the MCC is the best so far
        if avg_mcc > test_t_mcc:
            test_t_mcc = avg_mcc
            torch.save(model.state_dict(), 'model_parameters/model.pt')


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


train()
predict()