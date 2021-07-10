import pickle
import numpy as np
import torch
from torch.utils import data

from models.LocalCNN import CNNPPI
from models.LocalCNNAttention import AttentionPPI
from data_generator import ProteinDataSet
from evaluation import compute_roc, compute_aupr, compute_mcc,acc_score, compute_performance
from config import DefaultConfig
configs = DefaultConfig()


def test(model, loader, path_dir):
    # Model on eval mode
    model.eval()
    result = []
    all_trues = []

    for batch_idx, (seq_data, pssm_data, dssp_data, rasa_data, angle_data, local_data, label) in enumerate(loader):
        # Create variables
        with torch.no_grad():
            if torch.cuda.is_available():
                seq_var = seq_data.cuda().float()
                pssm_var = pssm_data.cuda().float()
                dssp_var = dssp_data.cuda().float()
                rasa_var = rasa_data.cuda().float()
                angle_var = angle_data.cuda().float()
                local_var = local_data.cuda().float()
            else:
                seq_var = seq_data.float()
                pssm_var = pssm_data.float()
                dssp_var = dssp_data.float()
                rasa_var = rasa_data.float()
                angle_var = angle_data.float()
                local_var = local_data.float()

        # compute output
        output = model(local_var)
        # print(output)
        shapes = output.data.shape
        output = output.view(shapes[0]*shapes[1])
        result.append(output.data.cpu().numpy())
        all_trues.append(label.numpy())
        # g_auc_roc = compute_roc(output.data.cpu().numpy(), label.numpy())
        # print(g_auc_roc)

    # calculate
    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(result, axis=0)
    # print(all_preds.shape, all_trues.shape)
    auc = compute_roc(all_preds, all_trues)
    aupr = compute_aupr(all_preds, all_trues)
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_preds,all_trues)
    acc = acc_score(predictions_max,all_trues)
    mcc = compute_mcc(predictions_max, all_trues)

    print('acc:%0.6f,F_value:%0.6f, precision:%0.6f,recall:%0.6f,auc:%0.6f,aupr:%0.6f,mcc:%0.6f,threadhold:%0.6f\n' % (
        acc, f_max, p_max, r_max, auc, aupr, mcc, t_max))

    predict_result = {}
    predict_result["pred"] = all_preds
    predict_result["label"] = all_trues
    result_file = "{0}/test_predict.pkl".format(path_dir)
    with open(result_file, "wb") as fp:
        pickle.dump(predict_result, fp)


if __name__ == '__main__':
    # parameters
    batch_size = configs.batch_size
    test_data_sets = configs.data_sets
    save_path = configs.save_path

    test_sequences_file = ['data_cache/{0}_sequence_data.pkl'.format(key) for key in test_data_sets]
    test_dssp_file = ['data_cache/{0}_dssp2_data.pkl'.format(key) for key in test_data_sets]
    test_pssm_file = ['data_cache/{0}_pssm_data.pkl'.format(key) for key in test_data_sets]
    test_rasa_file = ['data_cache/{0}_rasa_data.pkl'.format(key) for key in test_data_sets]
    test_angle_file = ['data_cache/{0}_angle_data.pkl'.format(key) for key in test_data_sets]
    test_label_file = ['data_cache/{0}_label.pkl'.format(key) for key in test_data_sets]
    all_list_file = 'data_cache/all_dset_list.pkl'
    test_list_file = 'data_cache/testing_list.pkl'

    # Dataset
    test_dataset = ProteinDataSet(test_sequences_file, test_pssm_file, test_dssp_file, test_rasa_file, test_angle_file, test_label_file, all_list_file)
    with open(test_list_file, "rb") as fp:
        test_list = pickle.load(fp)
    test_samples = data.sampler.SubsetRandomSampler(test_list)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_samples,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0, drop_last=False)

    # Model
    model_file = "{0}/AttentionPPI_model{1}.dat".format(save_path, 1)
    class_nums = 1
    # model = CNNPPI(class_nums)
    model = AttentionPPI(class_nums)
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    test(model, test_loader, save_path)
    print('Done!')