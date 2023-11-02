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


class ComplexBatchNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, complex_axis=1):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features // 2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br', None)
            self.register_parameter('Bi', None)

        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(self.num_features))
            self.register_buffer('RMi', torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones(self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr', None)
            self.register_parameter('RMi', None)
            self.register_parameter('RVrr', None)
            self.register_parameter('RVri', None)
            self.register_parameter('RVii', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert (xr.shape == xi.shape)
        assert (xr.size(1) == self.num_features)

    def forward(self, inputs):
        # self._check_input_dim(xr, xi)

        xr, xi = torch.chunk(inputs, 2, axis=self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (- Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs


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

    if s1.shape[1] > s2.shape[1]:
        s1 = s1[:, :s2.shape[1]]
    else:
        s2 = s2[:, :s1.shape[1]]

    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)

    s_target = s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)


class NavieComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, batch_first=False):
        super(NavieComplexLSTM, self).__init__()

        self.input_dim = input_size // 2
        self.rnn_units = hidden_size // 2

        self.real_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional,
                                 batch_first=False)
        self.imag_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional,
                                 batch_first=False)

    def forward(self, inputs):
        if isinstance(inputs, list):
            real, imag = inputs
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, -1)
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out
        return [real_out, imag_out]

    def flatten_parameters(self):
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()


class DCCRN(nn.Module):
    def __init__(self, config):
        super(DCCRN, self).__init__()
        self.config = config

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.tanh = nn.Tanh()

        # encoder
        for channel_idx in range(len(config['conv_channels'])-1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(in_channels=config['conv_channels'][channel_idx],
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

        # decoder
        for channel_idx in range(len(config['conv_channels']) - 1, 0, -1):
            if channel_idx!=1:
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
        self.lstm = nn.LSTM(input_size=config['conv_channels'][-1] * 4, hidden_size=config['conv_channels'][-1] * 4, num_layers=1)

        # stft
        self.stft = ConvSTFT(self.win_len, self.win_inc, self.win_len, self.win_type, 'complex', fix=True)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.win_len, self.win_type, 'complex', fix=True)

    def forward(self, mic, far, near, echo):

        encoder_out = []

        # stft
        mic_specs = self.stft(mic)
        far_specs = self.stft(far)
        near_specs = self.stft(near)
        echo_specs = self.stft(echo)

        # split real and imag
        mic_real = mic_specs[:, :self.win_len // 2 + 1]
        mic_imag = mic_specs[:, self.win_len // 2 + 1:]

        far_real = far_specs[:, :self.win_len // 2 + 1]
        far_imag = far_specs[:, self.win_len // 2 + 1:]

        near_real = near_specs[:, :self.win_len // 2 + 1]
        near_imag = near_specs[:, self.win_len // 2 + 1:]

        echo_real = echo_specs[:, :self.win_len // 2 + 1]
        echo_imag = echo_specs[:, self.win_len // 2 + 1:]


        cIRM_r = (mic_real * near_real + mic_imag * near_imag) / (mic_real ** 2 + mic_imag ** 2 + 1e-9)
        cIRM_i = (mic_real * near_imag - mic_imag * near_real) / (mic_real ** 2 + mic_imag ** 2 + 1e-9)


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

        loss_r = F.mse_loss(mask_real, cIRM_r, reduction='mean')
        loss_i = F.mse_loss(mask_imag, cIRM_i, reduction='mean')
        loss_mask = loss_r + loss_i

        real, imag = mic_real * mask_real - mic_imag * mask_imag, mic_real * mask_imag + mic_imag * mask_real

        echor, echoi = echo_real * mask_real - echo_imag * mask_imag, echo_real * mask_imag + echo_imag * mask_real
        loss_echo = F.mse_loss(echor, torch.zeros_like(echor), reduction='mean') + F.mse_loss(echoi, torch.zeros_like(echoi), reduction='mean')

        loss = 0.3*loss_mask + 0.7*loss_echo

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)

        return out_wav, out_spec, near_specs, loss





