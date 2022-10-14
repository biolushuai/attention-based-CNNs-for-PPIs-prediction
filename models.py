import torch

from layers import *
from configs import DefaultConfig
configs = DefaultConfig()


class AttCNNModel(nn.Module):
    def __init__(self):
        super(AttCNNModel, self).__init__()
        mlp_dim = configs.mlp_dim
        dropout_rate = configs.dropout_rate
        cnn_in_dim = configs.feature_dim * 2
        mlp_in_dim = configs.out_channel * 3

        self.attention = AttentionLayer()
        self.vertex2image = VertexToImage()
        self.cnn = ConvolutionLayer(cnn_in_dim)

        self.linear1 = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, indices):
        vertex, attention_score = self.attention(vertex)
        vertex = self.vertex2image(vertex)
        out = torch.unsqueeze(vertex, 1)
        out = out[indices]
        out = self.cnn(out)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

    def get_attention_score(self, v):
        _, a_score = self.attention(v)
        return a_score


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        mlp_dim = configs.mlp_dim
        dropout_rate = configs.dropout_rate
        cnn_in_dim = configs.feature_dim
        mlp_in_dim = configs.out_channel * 3

        self.vertex2image = VertexToImage()
        self.cnn = ConvolutionLayer(cnn_in_dim)

        self.linear1 = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, indices):
        vertex = self.vertex2image(vertex)
        out = torch.unsqueeze(vertex, 1)
        out = out[indices]
        out = self.cnn(out)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


if __name__ == '__main__':
    in_data = torch.rand([151, 52])
    in_ids = torch.randint(0, 150, [32])
    # m = CNNModel()
    m = AttCNNModel()
    a_s, pi = m.get_attention_score(in_data)
    print(a_s.shape, pi.shape)