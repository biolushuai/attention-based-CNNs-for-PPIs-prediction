"""
This code is from and https://github.com/tcxdgit/cnn_multilabel_classification
"""

import pickle
import numpy as np
import torch
import torch.utils.data.sampler as sampler
from torch.utils import data
from config import DefaultConfig
configs = DefaultConfig()


class ProteinDataSet(data.Dataset):
    def __init__(self, sequence_file=None, pssm_file=None, dssp2_file=None, rasa_file=None, angle_file=None, label_file=None, list_file=None):
        super(ProteinDataSet, self).__init__()
        
        self.all_sequences = []
        for seq_file in sequence_file:
            with open(seq_file, "rb") as fp_seq:
               temp_seq = pickle.load(fp_seq)
            self.all_sequences.extend(temp_seq)

        self.all_pssm = []
        for pm_file in pssm_file: 
            with open(pm_file, "rb") as fp_pssm:
                temp_pssm = pickle.load(fp_pssm)
            self.all_pssm.extend(temp_pssm)

        # self.all_dssp = []
        # for dp_file in dssp_file:
        #     with open(dp_file, "rb") as fp_dssp:
        #         temp_dssp = pickle.load(fp_dssp)
        #     self.all_dssp.extend(temp_dssp)

        self.all_dssp2 = []
        for dp2_file in dssp2_file:
            with open(dp2_file, "rb") as fp_dssp2:
                temp_dssp2 = pickle.load(fp_dssp2)
            self.all_dssp2.extend(temp_dssp2)

        self.all_rasa = []
        for asa_file in rasa_file:
            with open(asa_file, "rb") as fp_asa:
                temp_sa = pickle.load(fp_asa)
            self.all_rasa.extend(temp_sa)

        self.all_angle = []
        for ang_file in angle_file:
            with open(ang_file, "rb") as fp_ang:
                temp_ang = pickle.load(fp_ang)
            self.all_angle.extend(temp_ang)

        self.all_label = []
        for lab_file in label_file: 
            with open(lab_file, "rb") as fp_label:
                temp_label = pickle.load(fp_label)
            self.all_label.extend(temp_label)

        with open(list_file, "rb") as list_label:
            self.protein_list = pickle.load(list_label)

        self.Config = DefaultConfig()
        self.max_seq_len = self.Config.max_sequence_length
        self.window_size = self.Config.window_size

    def __getitem__(self, index):
        count, id_idx, ii, data_set, protein_id, seq_length = self.protein_list[index]
        window_size = self.window_size
        id_idx = int(id_idx)
        win_start = ii - window_size
        win_end = ii + window_size
        seq_length = int(seq_length)
        label_idx = (win_start + win_end)//2
        
        all_seq_features = []
        seq_len = 0
        for idx in self.all_sequences[id_idx][:self.max_seq_len]:
            acid_one_hot = [0 for _ in range(20)]
            acid_one_hot[idx] = 1
            all_seq_features.append(acid_one_hot)
            seq_len += 1
        while seq_len < self.max_seq_len:
            acid_one_hot = [0 for _ in range(20)]
            all_seq_features.append(acid_one_hot)
            seq_len += 1

        all_pssm_features = self.all_pssm[id_idx][:self.max_seq_len]
        seq_len = len(all_pssm_features)
        while seq_len < self.max_seq_len:
            zero_vector = [0 for _ in range(20)]
            all_pssm_features.append(zero_vector)
            seq_len += 1

        # all_dssp_features = self.all_dssp[id_idx][:self.max_seq_len]
        # seq_len = len(all_dssp_features)
        # while seq_len < self.max_seq_len:
        #     zero_vector = [0 for _ in range(9)]
        #     all_dssp_features.append(zero_vector)
        #     seq_len += 1

        all_dssp2_features = self.all_dssp2[id_idx][:self.max_seq_len]
        seq_len = len(all_dssp2_features)
        while seq_len < self.max_seq_len:
            zero_vector = [0 for _ in range(8)]
            all_dssp2_features.append(zero_vector)
            seq_len += 1

        all_rasa_features = self.all_rasa[id_idx][:self.max_seq_len]
        seq_len = len(all_rasa_features)
        while seq_len < self.max_seq_len:
            zero_vector = [0 for _ in range(2)]
            all_rasa_features.append(zero_vector)
            seq_len += 1

        all_angle_features = self.all_angle[id_idx][:self.max_seq_len]
        seq_len = len(all_angle_features)
        while seq_len < self.max_seq_len:
            zero_vector = [0 for _ in range(2)]
            all_angle_features.append(zero_vector)
            seq_len += 1

        local_features = []
        # labels = []
        while win_start < 0:
            data = []
            acid_one_hot = [0 for _ in range(20)]
            data.extend(acid_one_hot)

            pssm_zero_vector = [0 for _ in range(20)]
            data.extend(pssm_zero_vector)

            # dssp_zero_vector = [0 for _ in range(9)]
            # data.extend(dssp_zero_vector)

            dssp2_zero_vector = [0 for _ in range(8)]
            data.extend(dssp2_zero_vector)

            rasa_zero_vector = [0 for _ in range(2)]
            data.extend(rasa_zero_vector)

            angle_zero_vector = [0 for _ in range(2)]
            data.extend(angle_zero_vector)

            local_features.extend(data)
            win_start += 1
       
        valid_end = min(win_end, seq_length-1)
        while win_start <= valid_end:
            data = []
            idx = self.all_sequences[id_idx][win_start]

            acid_one_hot = [0 for _ in range(20)]
            acid_one_hot[idx] = 1
            data.extend(acid_one_hot)

            pssm_val = self.all_pssm[id_idx][win_start]
            data.extend(pssm_val)

            # try:
            #     dssp_val = self.all_dssp[id_idx][win_start]
            # except:
            #     dssp_val = [0 for _ in range(9)]
            # data.extend(dssp_val)

            dssp2_val = self.all_dssp2[id_idx][win_start]
            data.extend(dssp2_val)

            rasa_val = self.all_rasa[id_idx][win_start]
            data.extend(rasa_val)

            angle_val = self.all_angle[id_idx][win_start]
            data.extend(angle_val)

            local_features.extend(data)
            win_start += 1

        while win_start <= win_end:
            data = []
            acid_one_hot = [0 for _ in range(20)]
            data.extend(acid_one_hot)

            pssm_zero_vector = [0 for _ in range(20)]
            data.extend(pssm_zero_vector)

            # dssp_zero_vector = [0 for _ in range(9)]
            # data.extend(dssp_zero_vector)

            dssp2_zero_vector = [0 for _ in range(8)]
            data.extend(dssp2_zero_vector)

            rasa_zero_vector = [0 for _ in range(2)]
            data.extend(rasa_zero_vector)

            angle_zero_vector = [0 for _ in range(2)]
            data.extend(angle_zero_vector)

            local_features.extend(data)
            win_start += 1

        label = self.all_label[id_idx][label_idx]
        label = np.array(label, dtype=np.float32)

        all_seq_features = np.stack(all_seq_features)
        all_seq_features = all_seq_features[np.newaxis, :, :]
        all_pssm_features = np.stack(all_pssm_features)
        all_pssm_features = all_pssm_features[np.newaxis, :, :]
        # all_dssp_features = np.stack(all_dssp_features)
        # all_dssp_features = all_dssp_features[np.newaxis, :, :]
        all_dssp2_features = np.stack(all_dssp2_features)
        all_dssp2_features = all_dssp2_features[np.newaxis, :, :]
        all_rasa_features = np.stack(all_rasa_features)
        all_rasa_features = all_rasa_features[np.newaxis, :, :]
        all_angle_features = np.stack(all_angle_features)
        all_angle_features = all_angle_features[np.newaxis, :, :]

        local_features = np.stack(local_features)
        local_features = local_features.reshape(1, -1, 52)
        # print(local_features.shape)

        return all_seq_features, all_pssm_features, all_dssp2_features, all_rasa_features, all_angle_features, local_features, label

    def __len__(self):
        return len(self.protein_list)


if __name__ == '__main__':
    train_data = ["dset186", "dset164", "dset72"]
    train_sequences_file = ['data_cache/{0}_sequence_data.pkl'.format(key) for key in train_data]
    train_dssp_file = ['data_cache/{0}_dssp2_data.pkl'.format(key) for key in train_data]
    train_pssm_file = ['data_cache/{0}_pssm_data.pkl'.format(key) for key in train_data]
    train_rasa_file = ['data_cache/{0}_rasa_data.pkl'.format(key) for key in train_data]
    train_angle_file = ['data_cache/{0}_angle_data.pkl'.format(key) for key in train_data]
    train_label_file = ['data_cache/{0}_label.pkl'.format(key) for key in train_data]
    all_list_file = 'data_cache/all_dset_list.pkl'
    train_list_file = 'data_cache/training_list.pkl'

    batch_size = configs.batch_size
    train_data_set = ProteinDataSet(train_sequences_file, train_pssm_file, train_dssp_file, train_rasa_file, train_angle_file, train_label_file, all_list_file)

    with open(train_list_file, "rb") as fp:
        train_list = pickle.load(fp)
        train_list = train_list[:10]

    samples_num = len(train_list)
    split_num = int(configs.split_rate * samples_num)
    data_index = train_list
    np.random.shuffle(data_index)
    train_index = data_index[:split_num]
    train_samples = sampler.SubsetRandomSampler(train_index)

    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, sampler=train_samples,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0, drop_last=False)

    for i, data in enumerate(train_loader):
        for d in data:
            print(d.shape)