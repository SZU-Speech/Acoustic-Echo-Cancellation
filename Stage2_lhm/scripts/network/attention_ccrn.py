import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))


class ConvSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__()

        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase


class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConviSTFT, self).__init__()
        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """

        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        # outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[..., self.win_len - self.stride:-(self.win_len - self.stride)]

        return outputs

class ComplexConv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
    ):
        '''
            in_channels: real+imag
            out_channels: real+imag
            kernel_size : input [B,C,D,T] kernel size in [D,T]
            padding : input [B,C,D,T] padding in [D,T]
            causal: if causal, will padding time dimension's left side,
                    otherwise both

        '''
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.complex_axis = 1
        self.groups = groups
        self.dilation = dilation
        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)

        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)

        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)
        return out


class ComplexConvTranspose2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            complex_axis=1,
            groups=1
    ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(ComplexConvTranspose2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.real_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, output_padding=output_padding, groups=self.groups)
        self.imag_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, output_padding=output_padding, groups=self.groups)
        self.complex_axis = complex_axis

        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs):

        real, imag = torch.chunk(inputs, 2, self.complex_axis)

        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)

        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


def complex_cat(inputs, axis):
    real, imag = [], []
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data, 2, axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real, axis)
    imag = torch.cat(imag, axis)
    outputs = torch.cat([real, imag], axis)
    return outputs


# l2范数
def l2_norm(s1, s2):
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)


class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ATT_CCRN(nn.Module):
    def __init__(self, config):
        super(ATT_CCRN, self).__init__()
        self.config = config

        self.mic_encoder = nn.ModuleList()
        self.far_encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.tanh = nn.Tanh()

        # encoder
        for channel_idx in range(len(config['conv_channels'])-1):
            self.far_encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=config['conv_channels'][channel_idx],
                              out_channels=config['conv_channels'][channel_idx + 1],
                              kernel_size=config['kernel_size'],
                              stride=config['stride'],
                              padding=config['padding'],
                              dilation=config['dilation'],
                              groups=config['groups']),
                    nn.BatchNorm2d(config['conv_channels'][channel_idx + 1]),
                    nn.ReLU(),
                )
            )
            if channel_idx == 0:
                self.mic_encoder.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=config['conv_channels'][channel_idx],
                                  out_channels=config['conv_channels'][channel_idx+1],
                                  kernel_size=config['kernel_size'],
                                  stride=config['stride'],
                                  padding=config['padding'],
                                  dilation=config['dilation'],
                                  groups=config['groups']),
                        nn.BatchNorm2d(config['conv_channels'][channel_idx + 1]),
                        nn.PReLU(),
                    )
                )
            else:
                self.mic_encoder.append(
                    nn.Sequential(
                        ComplexConv2d(in_channels=config['conv_channels'][channel_idx],
                                  out_channels=config['conv_channels'][channel_idx + 1],
                                  kernel_size=config['kernel_size'],
                                  stride=config['stride'],
                                  padding=config['padding'],
                                  dilation=config['dilation'],
                                  groups=config['groups']),
                        nn.BatchNorm2d(config['conv_channels'][channel_idx + 1]),
                        nn.PReLU(),
                    )
                )


        # decoder
        for channel_idx in range(len(config['conv_channels']) - 1, 0, -1):
            if channel_idx!=0:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            in_channels=config['conv_channels'][channel_idx]*2,
                            out_channels=config['conv_channels'][channel_idx-1],
                            kernel_size=config['kernel_size'],
                            stride=config['stride'],
                            padding=config['padding'],
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(config['conv_channels'][channel_idx-1]),
                        nn.PReLU(),
                        )
                    )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            in_channels=config['conv_channels'][channel_idx] * 2,
                            # out_channels=config['conv_channels'][channel_idx - 1],
                            out_channels=2,
                            kernel_size=config['kernel_size'],
                            stride=config['stride'],
                            padding=config['padding'],
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(2),
                        nn.Tanh(),
                    )
                )

        self.win_type = 'hann'
        self.win_len = self.config['win_size']
        self.win_inc = self.config['hop_size']

        # LSTM
        self.lstm = nn.LSTM(input_size=config['conv_channels'][-1] * 8, hidden_size=config['conv_channels'][-1] * 8, num_layers=1)

        # stft
        self.real_stft = ConvSTFT(self.win_len, self.win_inc, self.win_len, self.win_type, 'real', fix=True)
        self.cpx_stft = ConvSTFT(self.win_len, self.win_inc, self.win_len, self.win_type, 'complex', fix=True)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.win_len, self.win_type, 'complex', fix=True)

    def forward(self, mic, far, near):

        encoder_out = []

        # stft
        mic_mag, _ = self.real_stft(mic)
        far_mag, _ = self.real_stft(far)
        near_specs = self.cpx_stft(near)

        # split real and imag
        near_real = near_specs[:, :self.win_len // 2 + 1]
        near_imag = near_specs[:, self.win_len // 2 + 1:]

        # stack
        cspecs = torch.stack((mic_real, far_real, mic_imag, far_imag), dim=1)
        out = cspecs[:, :, 1:]

        # encoder
        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            encoder_out.append(out)

        # rnn
        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        out = torch.reshape(out, [lengths, batch_size, channels * dims])
        out, _ = self.lstm(out)
        out = torch.reshape(out, [lengths, batch_size, channels, dims])
        out = out.permute(1, 2, 3, 0)

        # decoder
        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)

        # output
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

        real, imag = mic_real * mask_real - mic_imag * mask_imag, mic_real * mask_imag + mic_imag * mask_real
        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)

        return out_wav, out_spec, near_specs





