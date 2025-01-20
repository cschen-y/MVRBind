import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, \
    precision_recall_curve
from torch_geometric.loader import DataLoader
from MVRBind.data_process.test_18_dataset import Test18Dataset
from model import MVRBind
from data_process.get_pdb_feature import TestDataset
from sklearn.metrics import auc

def predict():
    apo_dataset = TestDataset(root='./pt/apo_pt', pdb_file_path='data_process/data/apo_pdb',
                              fasta_file_path="data_process/data/apo_fastas",
                              msa_file_path='data_process/data/apo_msa_result',
                              em_file_path='data_process/data/apo_em', asa_file_path="data_process/data/apo_asa",
                              top_k=8, label_file_path="data_process/label/label_Tapo.pkl", mode="apo_test")
    test9_dataset = TestDataset(root=f'./pt/test9_pt', pdb_file_path='data_process/data/test9_pdb', fasta_file_path="data_process/data/test9_fastas", msa_file_path='data_process/data/test9_ag', em_file_path='data_process/data/test9_em', asa_file_path ="data_process/data/test9_asa", label_file_path="data_process/label/test9_label.pkl", top_k=8)
    test_18_dataset = Test18Dataset(f'./pt/kop8')
    test18_loder = DataLoader(test_18_dataset, batch_size=18)
    test9_loader = DataLoader(test9_dataset, batch_size=9)
    apo_loader = DataLoader(apo_dataset, batch_size=8)
    test_c_1 = TestDataset("./pt/conformational_pt/0","_","_","_","_","_","_","_","_")
    test_c_2 = TestDataset("./pt/conformational_pt/1", "_", "_", "_", "_","_","_","_","_")
    test_c_3 = TestDataset("./pt/conformational_pt/2", "_", "_", "_", "_","_","_","_","_")
    test_loader_1 = DataLoader(test_c_1, batch_size=30)
    test_loader_2 = DataLoader(test_c_2, batch_size=30)
    test_loader_3 = DataLoader(test_c_3, batch_size=30)
    model = MVRBind(136)
    model.load_state_dict(torch.load(f"model_parameters/model.pt"))
    model.eval()
    for data in test18_loder:
        output = model(data)
        threshold = 0.532
        binary_predictions = (output.detach().numpy() > threshold).astype(int)
        accuracy = accuracy_score(data.y, binary_predictions)
        precision = precision_score(data.y, binary_predictions)
        recall = recall_score(data.y, binary_predictions)
        f1 = f1_score(data.y, binary_predictions)
        mcc_value = matthews_corrcoef(data.y, binary_predictions)
        roc = roc_auc_score(data.y, output.detach().numpy())
    print(f'Test18: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc_value:.4f}, ROC AUC: {roc:.4f}')

    for data in test9_loader:
        output = model(data)
        threshold = 0.424
        binary_predictions = (output.detach().numpy() > threshold).astype(int)
        accuracy = accuracy_score(data.y, binary_predictions)
        precision = precision_score(data.y, binary_predictions)
        recall = recall_score(data.y, binary_predictions)
        f1 = f1_score(data.y, binary_predictions)
        mcc_value = matthews_corrcoef(data.y, binary_predictions)
        roc = roc_auc_score(data.y, output.detach().numpy())
        print(f'Test9: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc_value:.4f}, ROC AUC: {roc:.4f}')

    for data in apo_loader:
        output = model(data)
        threshold = 0.408
        binary_predictions = (output.detach().numpy() > threshold).astype(int)
        accuracy = accuracy_score(data.y, binary_predictions)
        precision = precision_score(data.y, binary_predictions)
        recall = recall_score(data.y, binary_predictions)
        f1 = f1_score(data.y, binary_predictions)
        mcc_value = matthews_corrcoef(data.y, binary_predictions)
        roc = roc_auc_score(data.y, output.detach().numpy())
        print(f'Apo_test: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc_value:.4f}, ROC AUC: {roc:.4f}')

    all_ground_truth = []
    all_predictions = []
    auc_values_combined = []
    all_predictions_pro = []
    for test_loader in [test_loader_1, test_loader_2, test_loader_3]:
        for data in test_loader:
            output = model(data)
            threshold = 0.390
            binary_predictions = (output.detach().numpy() > threshold).astype(int)
            all_ground_truth.extend(data.y.cpu().numpy())
            all_predictions.extend(binary_predictions)
            all_predictions_pro.extend(list(output.detach().numpy()))
            roc = roc_auc_score(data.y.cpu().numpy(), output.detach().cpu().numpy())
            auc_values_combined.append(roc)

    all_ground_truth = np.array(all_ground_truth)
    all_predictions = np.array(all_predictions)
    accuracy = accuracy_score(all_ground_truth, all_predictions)
    precision = precision_score(all_ground_truth, all_predictions)
    recall = recall_score(all_ground_truth, all_predictions)
    f1 = f1_score(all_ground_truth, all_predictions)
    mcc_value = matthews_corrcoef(all_ground_truth, all_predictions)
    roc = roc_auc_score(all_ground_truth, all_predictions_pro)

    print(f'Conformational_test: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc_value:.4f}, ROC AUC: {roc:.4f}')

