import os
import csv
import pickle
import torch
from models import *
from metrics import *
from utils import *
from configs import DefaultConfig
configs = DefaultConfig()


def test(model, test_graphs):
    model.eval()
    all_auc_roc = []
    all_auc_pr = []

    all_acc = []
    all_mcc = []
    all_f_score = []
    all_precision = []
    all_recall = []

    for test_g in test_graphs:
        if torch.cuda.is_available():
            test_ppi_vertex = torch.FloatTensor(test_g['seq_features']).cuda()
            test_ppi_label = torch.LongTensor(test_g['seq_labels'])
            test_ppi_indices = [index for index, _ in enumerate(test_g['seq_labels'])]
            test_ppi_indices = torch.LongTensor(test_ppi_indices).cuda()

        g_preds = model(test_ppi_vertex, test_ppi_indices)
        g_preds = g_preds.data.cpu().numpy()
        test_ppi_label = test_ppi_label.numpy()

        g_auc_pr = compute_auc_pr(test_ppi_label, g_preds)
        g_auc_roc = compute_auc_roc(test_ppi_label, g_preds)
        g_preds[g_preds >= 0.5] = 1
        g_preds[g_preds < 0.5] = 0

        g_acc = compute_acc(test_ppi_label, g_preds)
        g_mcc, g_recall, g_precision, g_f1_score, _, _ = compute_performace(test_ppi_label, g_preds)
        print(test_g['seq_id'], np.sum(g_preds == 1), len(test_g['seq_features']), g_precision, g_recall, g_mcc)

        all_acc.append(g_acc)
        all_auc_pr.append(g_auc_pr)
        all_auc_roc.append(g_auc_roc)
        all_recall.append(g_recall)
        all_precision.append(g_precision)
        all_mcc.append(g_mcc)
        all_f_score.append(g_f1_score)

    print(np.mean(all_acc), np.mean(all_auc_roc), np.mean(all_auc_pr), np.mean(all_mcc), np.mean(all_f_score), np.mean(all_precision), np.mean(all_recall))
    return [np.mean(all_acc), np.mean(all_auc_roc), np.mean(all_auc_pr), np.mean(all_mcc), np.mean(all_f_score), np.mean(all_precision), np.mean(all_recall)]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    test_path = configs.test_dataset_path

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # test_model = CNNModel()
    test_model = AttCNNModel()

    if torch.cuda.is_available():
        model = test_model.cuda()

    # from utils import process_data
    # process_data(test_data)

    with open(os.path.join(configs.save_path, 'results_v2_0.5.csv'), 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['acc', 'auc_roc', 'auc_pr', 'mcc', 'f_max', 'p_max', 'r_max', 't_max'])


    for i in range(5):
        print('experiment:', i + 1)
        test_model_path = os.path.join(configs.save_path, 'model{}.tar'.format(i + 1))

        test_model_sd = torch.load(test_model_path)
        test_model.load_state_dict(test_model_sd)
        if torch.cuda.is_available():
            model = test_model.cuda()

        experiment_results = test(test_model, test_data)

        with open(os.path.join(configs.save_path, 'results_v2_0.5.csv'), 'a+', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(experiment_results)