class DefaultConfig(object):
    acid_one_hot = [0 for i in range(20)]
    acid_idx = {j: i for i, j in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    acid_idx2 = {i: j for i, j in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    BASE_PATH = "../../"
    sequence_path = "{0}/data_cache/sequence_data".format(BASE_PATH)
    pssm_path = "{0}/data_cache/pssm_data".format(BASE_PATH)
    dssp_path = "{0}/data_cache/dssp_data".format(BASE_PATH)
    dssp2_path = "{0}/data_cache/dssp2_data".format(BASE_PATH)
    rasa_path = "{0}/data_cache/rasa_data".format(BASE_PATH)
    angle_path = "{0}/data_cache/angle_data".format(BASE_PATH)

    save_path = r'D:\PythonProjects\InteractionSitePrediction\saved_models\ppi_feat52_local_cnnattention_t1_v5_e250_w5_b32_dp0.2_sp0.9'
    data_sets = ["dset186", "dset164", "dset72"]

    epochs = 250
    batch_size = 32
    learning_rate = 0.001
    feature_dim = 52
    dropout = 0.2
    split_rate = 0.9

    attention_dim = 56
    local_kernels = [3, 5, 7]
    window_size = 5
    out_channel = 64

    max_sequence_length = 500



