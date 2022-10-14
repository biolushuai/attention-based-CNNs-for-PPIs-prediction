import torch
from torch import nn
import torch.nn.functional as F

from configs import DefaultConfig
configs = DefaultConfig()


class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        input_dim = configs.feature_dim
        self.padding_size = configs.window_padding_size
        self.padding = nn.ZeroPad2d(padding=(0, 0, self.padding_size, self.padding_size))
        self.w = nn.Linear(input_dim, configs.attention_hidden_dim, bias=False)
        self.u = nn.Linear(input_dim, configs.attention_hidden_dim, bias=False)
        self.v = nn.Linear(configs.attention_hidden_dim, 1, bias=False)

    def forward(self, x):
        w_length = self.padding_size * 2
        x_length = len(x)
        x_center = x.unsqueeze(1).repeat(1, w_length, 1)

        x_padded = self.padding(x)
        x_local = list()
        for i in range(x_length):
            x_local.append(x_padded[i: i + self.padding_size])
            x_local.append(x_padded[i + self.padding_size + 1: i + self.padding_size * 2 + 1])
        x_local = torch.reshape(torch.cat(x_local, 0), [x_length, w_length, -1])

        energy = torch.tanh(torch.add(self.w(x_center), self.u(x_local)))
        attention = self.v(energy).squeeze(2)
        attention_out = F.softmax(attention, dim=1).unsqueeze(1)

        g = torch.bmm(attention_out, x_local).squeeze(1)
        out = torch.cat((g, x), dim=1)
        return out, torch.squeeze(attention_out)


class VertexToImage(nn.Module):
    def __init__(self):
        super(VertexToImage, self).__init__()
        self.padding_size = configs.window_padding_size
        self.padding = nn.ZeroPad2d(padding=(0, 0, self.padding_size, self.padding_size))

    def forward(self, x):
        w_length = self.padding_size * 2 + 1
        x_length = len(x)
        x_padded = self.padding(x)
        out = list()
        for i in range(x_length):
            out.append(x_padded[i: i + self.padding_size * 2 + 1])
        out = torch.reshape(torch.cat(out, 0), [x_length, w_length, -1])
        return out


class ConvolutionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(ConvolutionLayer, self).__init__()
        self.in_channel = 1
        self.out_channel = configs.out_channel
        self.kernels = configs.kernels
        self.feature_dim = feature_dim
        self.padding_size = configs.window_padding_size
        self.length = self.padding_size * 2 + 1
        self.padding = nn.ZeroPad2d(padding=(0, 0, self.padding_size, self.padding_size))

        padding1 = (self.kernels[0] - 1) // 2
        padding2 = (self.kernels[1] - 1) // 2
        padding3 = (self.kernels[2] - 1) // 2

        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1",
                              nn.Conv2d(self.in_channel, self.out_channel,
                                        padding=(padding1, 0),
                                        kernel_size=(self.kernels[0], self.feature_dim)))
        self.conv1.add_module("ReLU", nn.PReLU())
        self.conv1.add_module("pooling1", nn.MaxPool2d(kernel_size=(self.length, 1), stride=1))

        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2",
                              nn.Conv2d(self.in_channel, self.out_channel,
                                        padding=(padding2, 0),
                                        kernel_size=(self.kernels[1], self.feature_dim)))
        self.conv2.add_module("ReLU", nn.ReLU())
        self.conv2.add_module("pooling2", nn.MaxPool2d(kernel_size=(self.length, 1), stride=1))

        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3",
                              nn.Conv2d(self.in_channel, self.out_channel,
                                        padding=(padding3, 0),
                                        kernel_size=(self.kernels[2], self.feature_dim)))
        self.conv3.add_module("ReLU", nn.ReLU())
        self.conv3.add_module("pooling3", nn.MaxPool2d(kernel_size=(self.length, 1), stride=1))

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat((out1, out2, out3), 1)
        shapes = out.data.shape
        out = out.view(shapes[0], shapes[1])
        return out


if __name__ == '__main__':
    # in_data = torch.rand([32, 1, 13, 44])
    in_data2 = torch.rand([151, 52])
    # m = ConvolutionLayer(44)
    m = AttentionLayer()
    out_data, out_a = m(in_data2)
    print(out_data.shape, out_a.shape)
    print(out_a[:5])