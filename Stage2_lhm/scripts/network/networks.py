import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
# from torch_mfcc import MFCC, STFT, FBANK
# import torch_dct
from torch.nn import init
from torch.autograd import Variable
import numpy as np


# class MultiHeadSelfAttention(nn.Module):
#     dim_in: int  # input dimension
#     dim_k: int   # key and query dimension
#     dim_v: int   # value dimension
#     num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
#
#     def __init__(self, dim_in, dim_k, dim_v, num_heads=1):
#         super(MultiHeadSelfAttention, self).__init__()
#         assert dim_k % num_heads == 0 and dim_v % num_heads == 0
#
#         # QKV
#         self.dim_in = dim_in
#         self.dim_k = dim_k
#         self.dim_v = dim_v
#         self.num_heads = num_heads
#
#         # linear
#         self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
#         self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
#         self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
#
#         # activation
#         self.prelu_q = nn.PReLU()
#         self.prelu_k = nn.PReLU()
#         self.prelu_v = nn.PReLU()
#
#         self._norm_fact = 1 / sqrt(dim_k // num_heads)
#
#     def forward(self, mic, ref):
#         batch, n, dim_in = mic.shape
#         # assert dim_in == self.dim_in
#
#         nh = self.num_heads
#         dk = self.dim_k // nh  # dim_k of each head
#         dv = self.dim_v // nh  # dim_v of each head
#
#         q = self.prelu_q(self.linear_q(mic))
#         k = self.prelu_k(self.linear_k(ref))
#         v = self.prelu_v(self.linear_v(mic))
#
#         dist = q * k * self._norm_fact
#         dist = torch.softmax(dist, dim=2)
#
#         att = dist * v  # batch, nh, n, dv# batch, n, dim_v
#         return att


def Filpframe_OverlapA(x, win, inc):
    """
    基于重叠相加法的信号还原函数
    :param x: 分帧数据
    :param win: 窗
    :param inc: 帧移
    :return:
    """
    x = x.squeeze(1)
    batch_size, times, freqs = x.shape
    nx = (times - 1) * inc + freqs
    frameout = torch.zeros(batch_size, nx, device='cuda')
    for i in range(times):
        start = i * inc
        frameout[:, start:start + freqs] += x[:, i, :]
    return frameout

# class Self_Attention_AEC(nn.Module):
#     def __init__(self, config):
#         super(Self_Attention_AEC, self).__init__()
#         self.config = config
#
#         self.encoder = nn.ModuleList()
#         self.decoder = nn.ModuleList()
#         self.tanh = nn.Tanh()
#
#         # encoder
#         for channel_idx in range(len(config['conv_channels'])-1):
#             self.encoder.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels=config['conv_channels'][channel_idx],
#                               out_channels=config['conv_channels'][channel_idx+1],
#                               kernel_size=config['kernel_size'],
#                               stride=config['stride'],
#                               padding=config['padding']),
#                     nn.BatchNorm2d(config['conv_channels'][channel_idx + 1]),
#                     nn.PReLU(),
#                     # nn.Tanh(),
#                 )
#             )
#
#         # decoder
#         for channel_idx in range(len(config['conv_channels']) - 1, 0, -1):
#             if channel_idx!=1:
#                 self.decoder.append(
#                     nn.Sequential(
#                         nn.ConvTranspose2d(
#                             in_channels=config['conv_channels'][channel_idx]*2,
#                             out_channels=config['conv_channels'][channel_idx-1],
#                             kernel_size=config['kernel_size'],
#                             stride=config['stride'],
#                             padding=config['padding'],
#                             output_padding=(0, 1)
#                         ),
#                         nn.BatchNorm2d(config['conv_channels'][channel_idx-1]),
#                         nn.PReLU(),
#                         # nn.Tanh(),
#                         )
#                     )
#             else:
#                 self.decoder.append(
#                     nn.Sequential(
#                         nn.ConvTranspose2d(
#                             in_channels=config['conv_channels'][channel_idx] * 2,
#                             # out_channels=config['conv_channels'][channel_idx - 1],
#                             out_channels=1,
#                             kernel_size=config['kernel_size'],
#                             stride=config['stride'],
#                             padding=config['padding'],
#                             output_padding=(0, 0)
#                         ),
#                         # nn.BatchNorm2d(config['conv_channels'][channel_idx - 1]),
#                         nn.BatchNorm2d(1),
#                         nn.PReLU(),
#                         # nn.Tanh()
#                     )
#                 )
#
#         self.win = torch.hann_window(window_length=config['win_size']).cuda()
#         self.win_len = self.config['win_size']
#
#         # GRU
#         self.gru = nn.GRU(input_size=config['conv_channels'][-1] * 65, hidden_size=config['conv_channels'][-1] * 65, num_layers=2)
#
#         # init weights
#         self.initialize_weights()
#
#         # init dct
#         # self.initialize_dct()
#
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#                 nn.init.normal(m.weight, mean=0, std=1)
#                 # nn.init.kaiming_normal(m.weight, a=0, mode='fan_out',  nonlinearity='leaky_relu')
#                 # if m.bias is not None:
#                 #     m.init.constant_(m.bias, 0)
#             if isinstance(m, nn.GRU):
#                 nn.init.orthogonal(m.weight_ih_l0)
#                 nn.init.orthogonal(m.weight_hh_l0)
#
#     def initialize_dct(self):
#         m = torch.arange(1, self.config['win_size']+1, 1, device='cpu')
#         m = m.unsqueeze(0)
#
#         n = torch.arange(1, self.config['win_size'] + 1, 1, device='cpu')
#         n = n.unsqueeze(0).transpose(0, 1)
#
#         dn = torch.ones(self.config['win_size'], device='cpu')
#         dn[0] = torch.sqrt(torch.tensor(0.5, device='cpu'))
#
#         self.V = torch.cos(torch.mm((n-1), (2*m-1)) * torch.pi / (2*self.win_len)) * torch.sqrt(torch.tensor(2, device='cpu')/self.win_len)
#         self.U = torch.cos(torch.mm((n-1), (2*m-1)) * torch.pi / (2*self.win_len)) * torch.sqrt(torch.tensor(2, device='cpu')/self.win_len) * dn
#
#         self.V = torch.nn.parameter.Parameter(data=self.V, requires_grad=False)
#         self.U = torch.nn.parameter.Parameter(data=self.U, requires_grad=False)
#
#     def forward(self, mic, ref, near):
#
#         encoder_out = []
#
#         # unsqueeze
#         mic = mic.unsqueeze(1).unsqueeze(3)
#         ref = ref.unsqueeze(1).unsqueeze(3)
#         near = near.unsqueeze(1).unsqueeze(3)
#
#         # enframe
#         mic_unfold = F.unfold(input=mic, kernel_size=(self.config['win_size'], 1), stride=(self.config['hop_size'], 1), dilation=1, padding=0).transpose(1, 2)
#         ref_unfold = F.unfold(input=ref, kernel_size=(self.config['win_size'], 1), stride=(self.config['hop_size'], 1), dilation=1, padding=0).transpose(1, 2)
#         near_unfold = F.unfold(input=near, kernel_size=(self.config['win_size'], 1), stride=(self.config['hop_size'], 1), dilation=1, padding=0).transpose(1, 2)
#
#         # window
#         mic_unfold = self.win * mic_unfold
#         ref_unfold = self.win * ref_unfold
#         near_unfold = self.win * near_unfold
#
#         # dct
#         # mic_dct = mic_unfold @ self.U
#         # ref_dct = ref_unfold @ self.U
#         # near_dct = near_unfold @ self.U
#
#         mic_dct = torch_dct.dct(mic_unfold, norm=None)
#         ref_dct = torch_dct.dct(ref_unfold, norm=None)
#         near_dct = torch_dct.dct(near_unfold, norm=None)
#
#         # mic_dct = torch_dct.dct1(mic_unfold)
#         # ref_dct = torch_dct.dct1(ref_unfold)
#         # near_dct = torch_dct.dct1(near_unfold)
#
#         # suppress
#         mic_dct = self.tanh(mic_dct)
#         ref_dct = self.tanh(ref_dct)
#         near_dct = self.tanh(near_dct)
#
#         out = torch.stack([mic_dct, ref_dct], dim=1)
#
#         # encoder
#         for idx, layer in enumerate(self.encoder):
#             out = layer(out)
#             encoder_out.append(out)
#
#         # reshape
#         batch_size, channels, times, freqs = out.size()
#         out = out.transpose(1, 2).reshape(batch_size, times, channels*freqs)
#
#         # GRU
#         out = self.gru(out)[0]
#
#         # reshape
#         out = out.reshape(batch_size, times, channels, freqs)
#         out = out.transpose(1, 2)
#
#         # decoder
#         for idx in range(len(self.decoder)):
#             out = torch.cat([out, encoder_out[-1 - idx]], 1)
#             out = self.decoder[idx](out)
#
#         # idct
#         est_dct = torch_dct.idct(out.squeeze(1), norm=None)
#         # est_dct = out.squeeze(1) @ self.V
#
#         # Filpframe_Overlap
#         output = Filpframe_OverlapA(est_dct, self.config['win_size'], self.config['hop_size'])
#
#         return mic_dct, near_dct, output

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()

        self.config = config
        self.win = torch.hann_window(window_length=config['win_size']).cuda()
        self.win_len = self.config['win_size']
        self.hop_len = self.config['hop_size']
        self.step_list = torch.arange(0, self.config['win_size'], 1, device='cuda:0', dtype=torch.float32).unsqueeze(1)

        # self.nn = nn.Sequential(
        #
        #     # layer1
        #     nn.Linear(in_features=self.config['win_size'], out_features=self.config['win_size']//8, bias=True),
        #     # nn.Dropout(0.3),
        #     nn.PReLU(),
        #
        #     #layer3
        #     nn.Linear(in_features=self.config['win_size']//8, out_features=self.config['win_size']//8, bias=True),
        #     # nn.Dropout(0.3),
        #     nn.PReLU(),
        #
        #     # layer4
        #     nn.Linear(in_features=self.config['win_size']//8, out_features=self.config['win_size'], bias=True),
        #     # nn.Dropout(0.3),
        #     nn.Tanh(),
        # )
        self.nn = nn.Sequential(

            # layer1
            nn.Linear(in_features=100, out_features=100, bias=True),
            # nn.Dropout(0.3),
            nn.PReLU(),

            # layer3
            nn.Linear(in_features=100, out_features=100, bias=True),
            # nn.Dropout(0.3),
            nn.PReLU(),

            # layer4
            nn.Linear(in_features=100, out_features=100, bias=True),
            # nn.Dropout(0.3),
            nn.Tanh(),
        )

        k = torch.atleast_2d(torch.arange(0, self.config['win_size'], dtype=torch.float32))
        n = (torch.atleast_2d(torch.arange(0, self.config['win_size'], dtype=torch.float32)) + 1/2) * torch.pi / self.config['win_size']
        self.dct_matrix = torch.sqrt(2 / torch.tensor(self.config['win_size'], dtype=torch.float32)) * torch.cos(n.transpose(0, 1) @ k)
        self.dct_matrix[:, 0] = torch.sqrt(torch.tensor(1/2, dtype=torch.float32))*self.dct_matrix[:, 0]
        self.dct_matrix = self.dct_matrix.cuda()

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.sparse_(m.weight, sparsity=0.5, std=1)

    def forward(self, noisy, clean):

        noisy = noisy.unsqueeze(1).unsqueeze(3)
        clean = clean.unsqueeze(1).unsqueeze(3)

        noisy_enframe = F.unfold(input=noisy, kernel_size=(self.win_len, 1), stride=(self.hop_len, 1), dilation=1, padding=0)
        clean_enframe = F.unfold(input=clean, kernel_size=(self.win_len, 1), stride=(self.hop_len, 1), dilation=1, padding=0)

        noisy_enframe = noisy_enframe.transpose(1, 2)
        clean_enframe = clean_enframe.transpose(1, 2)

        noisy_win = noisy_enframe * self.win
        clean_win = clean_enframe * self.win

        noisy_dct = noisy_win @ self.dct_matrix
        clean_dct = clean_win @ self.dct_matrix

        # clamp
        noisy_dct = torch.clamp(input=noisy_dct, max=1, min=-1)
        clean_dct = torch.clamp(input=clean_dct, max=1, min=-1)

        # cut
        noisy_dct = noisy_dct[:, :, :100]
        clean_dct = clean_dct[:, :, :100]

        # nn
        output_dct = self.nn(noisy_dct)
        output_dct_pad = F.pad(output_dct, (0, self.config['win_size']-100), 'constant', 0)


        # idct
        output_idct = output_dct_pad @ self.dct_matrix.transpose(0, 1)

        #
        out_speech = Filpframe_OverlapA(output_idct, self.win_len, self.hop_len)

        # return output_dct, clean_dct, noisy_dct

        return output_dct, clean_dct, out_speech


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config

        self.win_len = self.config['win_size']
        self.hop_len = self.config['hop_size']

        self.win = torch.hann_window(window_length=config['win_size']).cuda()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # encoder
        for channel_idx in range(len(config['conv_channels'])-1):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=config['conv_channels'][channel_idx],
                              out_channels=config['conv_channels'][channel_idx+1],
                              kernel_size=config['kernel_size'],
                              stride=config['stride'],
                              padding=config['padding']),
                    nn.BatchNorm2d(config['conv_channels'][channel_idx + 1]),
                    nn.PReLU(),
                )
            )

        # decoder
        for channel_idx in range(len(config['conv_channels']) - 1, 0, -1):
            if channel_idx!=1:
                self.decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=config['conv_channels'][channel_idx]*2,
                            out_channels=config['conv_channels'][channel_idx-1],
                            kernel_size=config['kernel_size'],
                            stride=config['stride'],
                            padding=config['padding'],
                            output_padding=(0, 1)
                        ),
                        nn.BatchNorm2d(config['conv_channels'][channel_idx-1]),
                        nn.PReLU(),
                        # nn.Tanh(),
                        )
                    )
            else:
                self.decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=config['conv_channels'][channel_idx] * 2,
                            # out_channels=config['conv_channels'][channel_idx - 1],
                            out_channels=1,
                            kernel_size=config['kernel_size'],
                            stride=config['stride'],
                            padding=config['padding'],
                            output_padding=(0, 0)
                        ),
                        # nn.BatchNorm2d(config['conv_channels'][channel_idx - 1]),
                        nn.BatchNorm2d(1),
                        nn.Tanh()
                    )
                )
        k = torch.atleast_2d(torch.arange(0, self.config['win_size'], dtype=torch.float32))
        n = (torch.atleast_2d(torch.arange(0, self.config['win_size'], dtype=torch.float32)) + 1 / 2) * torch.pi / \
            self.config['win_size']
        self.dct_matrix = torch.sqrt(2 / torch.tensor(self.config['win_size'], dtype=torch.float32)) * torch.cos(
            n.transpose(0, 1) @ k)
        self.dct_matrix[:, 0] = torch.sqrt(torch.tensor(1 / 2, dtype=torch.float32)) * self.dct_matrix[:, 0]
        self.dct_matrix = self.dct_matrix.cuda()

    def forward(self, noisy, clean):

        # encoder_out = []
        clean_mag = torch.abs(torch.stft(clean, n_fft=512, return_complex=True))
        noisy_mag = torch.abs(torch.stft(noisy, n_fft=512, return_complex=True))
        noisy = noisy.unsqueeze(1).unsqueeze(3)
        clean = clean.unsqueeze(1).unsqueeze(3)

        noisy_enframe = F.unfold(input=noisy, kernel_size=(self.win_len, 1), stride=(self.hop_len, 1), dilation=1, padding=0)
        clean_enframe = F.unfold(input=clean, kernel_size=(self.win_len, 1), stride=(self.hop_len, 1), dilation=1, padding=0)

        noisy_enframe = noisy_enframe.transpose(1, 2)
        clean_enframe = clean_enframe.transpose(1, 2)

        noisy_win = noisy_enframe * self.win
        clean_win = clean_enframe * self.win

        noisy_dct = noisy_win @ self.dct_matrix
        clean_dct = clean_win @ self.dct_matrix



        # clamp
        # noisy_dct = torch.clamp(input=noisy_dct, max=1, min=-1)
        # clean_dct = torch.clamp(input=clean_dct, max=1, min=-1)

        # # cut
        # noisy_dct = noisy_dct[:, :, :100]
        # clean_dct = clean_dct[:, :, :100]
        #
        # # unsqueeze
        # noisy_dct = noisy_dct.unsqueeze(1)
        #
        # # encode features
        # for idx, layer in enumerate(self.encoder):
        #     out = layer(noisy_dct)
        #     encoder_out.append(out)
        #
        #     # reshape
        #     batch_size, channels, times, freqs = out.size()
        #     out = out.transpose(1, 2).reshape(batch_size, times, channels*freqs)
        #
        #     # GRU
        #     out = self.gru(out)[0]
        #
        #     # reshape
        #     out = out.reshape(batch_size, times, channels, freqs)
        #     out = out.transpose(1, 2)
        #
        #     # decoder
        #     for idx in range(len(self.decoder)):
        #         out = torch.cat([out, encoder_out[-1 - idx]], 1)
        #         out = self.decoder[idx](out)

        return noisy_dct, clean_dct, clean_mag,noisy_mag









