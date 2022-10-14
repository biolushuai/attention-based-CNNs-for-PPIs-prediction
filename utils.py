import numpy as np
from configs import DefaultConfig
configs = DefaultConfig()


def process_data(data):
    num_complex = len(data)
    for idx_complex in range(num_complex):
        # feat1 = data[idx_complex]['seq_feature'][:, :42]
        # feat2 = data[idx_complex]['seq_feature'][:, 50:]
        # data[idx_complex]['seq_feature'] = np.concatenate((feat1, feat2), 1)

        data[idx_complex]['seq_features'] = data[idx_complex]['seq_features'][:, :50]
