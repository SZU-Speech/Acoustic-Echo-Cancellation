# DCCRN
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.network.dccrn import ConvSTFT, ConviSTFT, ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, ComplexBatchNorm, complex_cat


class DCCRN(nn.Module):

    def __init__(self, config):
        super(DCCRN, self).__init__()
        self.config = config
        # for fft
        self.win_len = self.config['win_size']
        self.win_inc = self.config['hop_size']
        self.fft_len = self.config['win_size']
        self.win_type = self.config['win_type']

        self.kernel_num = self.config['conv_channels']
        self.use_cbn = self.config['use_cbn']

        self.use_clstm = self.config['use_clstm']
        self.rnn_layers = self.config['rnn_layers']
        self.rnn_units = self.config['rnn_units']

        self.masking_mode = self.config['masking_mode']

        # input_dim = win_len
        # output_dim = win_len

        # self.input_dim = input_dim
        # self.output_dim = output_dim
        # self.hidden_layers = rnn_layers
        # self.kernel_size = kernel_size

        #bidirectional=True
        bidirectional = False
        fac = 2 if bidirectional else 1

        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=True)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex', fix=True)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # encoder
        for channel_idx in range(len(config['conv_channels'])-1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(in_channels=config['conv_channels'][channel_idx],
                                  out_channels=config['conv_channels'][channel_idx + 1],
                                  kernel_size=config['kernel_size'],
                                  stride=config['stride'],
                                  padding=config['padding'],
                                  dilation=config['dilation'],
                                  groups=config['groups']),
                    nn.BatchNorm2d(self.kernel_num[channel_idx+1]) if not self.use_cbn else ComplexBatchNorm(self.kernel_num[channel_idx+1]),
                    nn.PReLU()
                )
            )

        # rnn
        hidden_dim = self.config['hidden_dim']
        # hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))
        if self.use_clstm:
            rnns = []
            for idx in range(self.rnn_layers):
                rnns.append(
                        NavieComplexLSTM(
                            input_size=hidden_dim * self.kernel_num[-1],
                            hidden_size=hidden_dim * self.kernel_num[-1],
                            bidirectional=bidirectional,
                            batch_first=False
                        )
                    )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.lstm = nn.LSTM(input_size=config['conv_channels'][-1] * hidden_dim, hidden_size=config['conv_channels'][-1] * hidden_dim, num_layers=self.rnn_layers)

        # decoder
        for channel_idx in range(len(self.kernel_num)-1, 0, -1):
            if channel_idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            in_channels=config['conv_channels'][channel_idx] * 2,
                            out_channels=config['conv_channels'][channel_idx - 1],
                            kernel_size=config['kernel_size'],
                            stride=config['stride'],
                            padding=config['padding'],
                            output_padding=(1, 0)
                        ),
                    nn.BatchNorm2d(self.kernel_num[channel_idx-1]) if not self.use_cbn else ComplexBatchNorm(self.kernel_num[channel_idx-1]),
                    nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            in_channels=config['conv_channels'][channel_idx] * 2,
                            out_channels=2,
                            kernel_size=config['kernel_size'],
                            stride=config['stride'],
                            padding=config['padding'],
                            output_padding=(1, 0)
                        ),
                    )
                )
        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, mic, far, near, echo):

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

        mic_mags = torch.sqrt(mic_real ** 2 + mic_imag ** 2 + 1e-8)
        mic_phase = torch.atan2(mic_imag, mic_real)

        # cIRM_r = (mic_real * near_real + mic_imag * near_imag) / (mic_real ** 2 + mic_imag ** 2 + 1e-9)
        # cIRM_i = (mic_real * near_imag - mic_imag * near_real) / (mic_real ** 2 + mic_imag ** 2 + 1e-9)

        cspecs = torch.stack([mic_real, far_real, mic_imag, far_imag], 1)
        cspecs = cspecs[:, :, 1:]

        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        '''

        out = cspecs
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            encoder_out.append(out)

        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        if self.use_clstm:
            r_rnn_in = out[:,:,:channels//2]
            i_rnn_in = out[:,:,channels//2:]
            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels//2*dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels//2*dims])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels//2, dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels//2, dims])
            out = torch.cat([r_rnn_in, i_rnn_in],2)

        else:
            # to [L, B, C, D]
            out = torch.reshape(out, [lengths, batch_size, channels*dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)

        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)

        mask_real = out[:,0]
        mask_imag = out[:,1]
        mask_real = F.pad(mask_real, [0,0,1,0])
        mask_imag = F.pad(mask_imag, [0,0,1,0])

        if self.masking_mode == 'E':
            mask_mags = (mask_real**2+mask_imag**2)**0.5
            real_phase = mask_real/(mask_mags+1e-8)
            imag_phase = mask_imag/(mask_mags+1e-8)
            mask_phase = torch.atan2(
                            imag_phase,
                            real_phase
                        )

            #mask_mags = torch.clamp_(mask_mags,0,100)
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags*mic_mags
            est_phase = mic_phase + mask_phase
            real = est_mags*torch.cos(est_phase)
            imag = est_mags*torch.sin(est_phase)
        elif self.masking_mode == 'C':
            real, imag = mic_real*mask_real-mic_imag*mask_imag, mic_real*mask_imag+mic_imag*mask_real
        elif self.masking_mode == 'R':
            real, imag = mic_real*mask_real, mic_imag*mask_imag

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)

        out_wav = torch.squeeze(out_wav, 1)
        return out_spec, out_wav, near_specs

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params