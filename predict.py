import numpy as np
import torch
import time
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from torch_geometric.loader import DataLoader
from data_process.test_18_dataset import Test18Dataset
from model import MVRBind
from data_process.get_pdb_feature import TestDataset
from sklearn.metrics import auc

def predict():
    apo_dataset = TestDataset(root='./pt/apo_pt')
    test_18_dataset = Test18Dataset(f'./pt/kop8')
    test_c_1 = TestDataset("./pt/conformational_pt/0")
    test_c_2 = TestDataset("./pt/conformational_pt/1")
    test_c_3 = TestDataset("./pt/conformational_pt/2")

    test18_loader = DataLoader(test_18_dataset, batch_size=18)
    apo_loader = DataLoader(apo_dataset, batch_size=30)
    test_loader_1 = DataLoader(test_c_1, batch_size=30)
    test_loader_2 = DataLoader(test_c_2, batch_size=30)
    test_loader_3 = DataLoader(test_c_3, batch_size=30)

    model = MVRBind(136)
    model.load_state_dict(torch.load("model_parameters/model.pt"))
    model.eval()

    with torch.no_grad():
        # -------- Test18 Dataset --------
        start_time = time.time()
        total_samples = 0
        for data in test18_loader:
            output = model(data)
            threshold = 0.6
            binary_predictions = (output.detach().numpy() > threshold).astype(int)
            accuracy = accuracy_score(data.y, binary_predictions)
            precision = precision_score(data.y, binary_predictions)
            recall = recall_score(data.y, binary_predictions)
            f1 = f1_score(data.y, binary_predictions)
            mcc_value = matthews_corrcoef(data.y, binary_predictions)
            roc = roc_auc_score(data.y, output.detach().numpy())
            total_samples += data.y.size(0)
        elapsed = time.time() - start_time
        print(f'Test18 inference time: {elapsed:.4f}s, Avg/sample: {elapsed / total_samples:.6f}s')
        print(f'Test18: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
              f'F1 Score: {f1:.4f}, MCC: {mcc_value:.4f}, ROC AUC: {roc:.4f}')

        # -------- Apo Dataset --------
        start_time = time.time()
        total_samples = 0
        for data in apo_loader:
            output = model(data)
            threshold = 0.23
            binary_predictions = (output.detach().numpy() > threshold).astype(int)
            accuracy = accuracy_score(data.y, binary_predictions)
            precision = precision_score(data.y, binary_predictions)
            recall = recall_score(data.y, binary_predictions)
            f1 = f1_score(data.y, binary_predictions)
            mcc_value = matthews_corrcoef(data.y, binary_predictions)
            roc = roc_auc_score(data.y, output.detach().numpy())
            total_samples += data.y.size(0)
        elapsed = time.time() - start_time
        print(f'Apo_test inference time: {elapsed:.4f}s, Avg/sample: {elapsed / total_samples:.6f}s')
        print(f'Apo_test: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
              f'F1 Score: {f1:.4f}, MCC: {mcc_value:.4f}, ROC AUC: {roc:.4f}')

        # -------- Conformational Test Sets --------
        all_ground_truth = []
        all_predictions = []
        auc_values_combined = []
        all_predictions_pro = []
        total_samples = 0
        start_time = time.time()
        for test_loader in [test_loader_1, test_loader_2, test_loader_3]:
            for data in test_loader:
                output = model(data)
                threshold = 0.48
                binary_predictions = (output.detach().numpy() > threshold).astype(int)
                all_ground_truth.extend(data.y.cpu().numpy())
                all_predictions.extend(binary_predictions)
                all_predictions_pro.extend(output.detach().numpy())
                roc = roc_auc_score(data.y.cpu().numpy(), output.detach().cpu().numpy())
                auc_values_combined.append(roc)
                total_samples += data.y.size(0)
        elapsed = time.time() - start_time

        all_ground_truth = np.array(all_ground_truth)
        all_predictions = np.array(all_predictions)
        accuracy = accuracy_score(all_ground_truth, all_predictions)
        precision = precision_score(all_ground_truth, all_predictions)
        recall = recall_score(all_ground_truth, all_predictions)
        f1 = f1_score(all_ground_truth, all_predictions)
        mcc_value = matthews_corrcoef(all_ground_truth, all_predictions)
        roc = roc_auc_score(all_ground_truth, all_predictions_pro)

        print(f'Conformational_test inference time: {elapsed:.4f}s, Avg/sample: {elapsed / total_samples:.6f}s')
        print(f'Conformational_test: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
              f'F1 Score: {f1:.4f}, MCC: {mcc_value:.4f}, ROC AUC: {roc:.4f}')


if __name__ == "__main__":
    predict()
