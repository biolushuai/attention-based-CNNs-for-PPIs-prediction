class DefaultConfig(object):
    train_dataset_path = r'./data/deepppisp_test.pkl'
    test_dataset_path = r'./data/pipenn_zk448_a_52.pkl'
    # test_dataset_path = r'./data/deepppisp_test_52.pkl'
    save_path = r'./models_saved/pipenn_a_feat52_att32_cnn64_k357_w5_e100_c10_b128_lr0.001_dp0.2_ng0.1_sp0.8'

    epochs = 100
    batch_size = 32

    learning_rate = 0.001
    weight_decay = 5e-4
    dropout_rate = 0.2
    # neg_wt = 0.06
    split_rate = 0.8

    #  input layer
    feature_dim = 52
    window_padding_size = 2

    # attention layer
    attention_hidden_dim = 32

    # cnn layer
    kernels = [3, 5, 7]
    out_channel = 64

    # mlp
    mlp_dim = 512



