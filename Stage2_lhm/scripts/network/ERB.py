import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network.attention_ccrn import ConvSTFT, ConviSTFT
from network.dccrn import si_snr


class EquivalentRectangularBandwidth():
    def __init__(self, nfreqs, sample_rate, total_erb_bands, low_freq, max_freq):
        if low_freq == None:
            low_freq = 20
        if max_freq == None:
            max_freq = sample_rate // 2
        freqs = np.linspace(0, max_freq, nfreqs)  # 每个STFT频点对应多少Hz
        self.EarQ = 9.265  # _ERB_Q
        self.minBW = 24.7  # minBW
        # 在ERB刻度上建立均匀间隔
        erb_low = self.freq2erb(low_freq)  # 最低 截止频率
        erb_high = self.freq2erb(max_freq)  # 最高 截止频率
        # 在ERB频率上均分为（total_erb_bands + ）2个 频带
        erb_lims = np.linspace(erb_low, erb_high, total_erb_bands + 2)
        cutoffs = self.erb2freq(erb_lims)  # 将 ERB频率再转到 hz频率, 在线性频率Hz上找到ERB截止频率对应的频率
        # self.nfreqs  F
        # self.freqs # 每个STFT频点对应多少Hz
        self.filters = self.get_bands(total_erb_bands, nfreqs, freqs, cutoffs)

    def freq2erb(self, frequency):
        """ [Hohmann2002] Equation 16"""
        return self.EarQ * np.log(1 + frequency / (self.minBW * self.EarQ))

    def erb2freq(self, erb):
        """ [Hohmann2002] Equation 17"""
        return (np.exp(erb / self.EarQ) - 1) * self.minBW * self.EarQ

    def get_bands(self, total_erb_bands, nfreqs, freqs, cutoffs):
        """
        获取erb bands、索引、带宽和滤波器形状
        :param erb_bands_num: ERB 频带数
        :param nfreqs: 频点数 F
        :param freqs: 每个STFT频点对应多少Hz
        :param cutoffs: 中心频率 Hz
        :param erb_points: ERB频带界限 列表
        :return:
        """
        cos_filts = np.zeros([nfreqs, total_erb_bands])  # (F, ERB)
        for i in range(total_erb_bands):
            lower_cutoff = cutoffs[i]  # 上限截止频率 Hz
            higher_cutoff = cutoffs[i + 2]  # 下限截止频率 Hz, 相邻filters重叠50%

            lower_index = np.min(np.where(freqs > lower_cutoff))  # 下限截止频率对应的Hz索引 Hz。np.where 返回满足条件的索引
            higher_index = np.max(np.where(freqs < higher_cutoff))  # 上限截止频率对应的Hz索引
            avg = (self.freq2erb(lower_cutoff) + self.freq2erb(higher_cutoff)) / 2
            rnge = self.freq2erb(higher_cutoff) - self.freq2erb(lower_cutoff)
            cos_filts[lower_index:higher_index + 1, i] = np.cos(
                (self.freq2erb(freqs[lower_index:higher_index + 1]) - avg) / rnge * np.pi)  # 减均值，除方差

        # 加入低通和高通，得到完美的重构
        filters = np.zeros([nfreqs, total_erb_bands + 2])  # (F, ERB)
        filters[:, 1:total_erb_bands + 1] = cos_filts

        # 低通滤波器上升到第一个余cos filter的峰值
        higher_index = np.max(np.where(freqs < cutoffs[1]))  # 上限截止频率对应的Hz索引
        filters[:higher_index + 1, 0] = np.sqrt(1 - np.power(filters[:higher_index + 1, 1], 2))

        # 高通滤波器下降到最后一个cos filter的峰值
        lower_index = np.min(np.where(freqs > cutoffs[total_erb_bands]))
        filters[lower_index:nfreqs, total_erb_bands + 1] = np.sqrt(
            1 - np.power(filters[lower_index:nfreqs, total_erb_bands], 2))
        return cos_filts


class TwoLayerGRUNet(nn.Module):
    def __init__(self, conf, erb_bands):
        super(TwoLayerGRUNet, self).__init__()

        self.config = conf
        self.win_len = self.config['win_size']
        self.win_inc = self.config['hop_size']
        self.win_type = 'hann'

        # 定义第一层GRU
        self.gru1 = nn.GRU(2 * erb_bands, 2*erb_bands, num_layers=1, batch_first=True, bias=True)

        # linear
        self.linear1 = nn.Linear(2*erb_bands, erb_bands, bias=True)
        self.linear2 = nn.Linear(erb_bands, erb_bands, bias=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # FFT
        self.cpx_stft = ConvSTFT(self.win_len, self.win_inc, self.win_len, self.win_type, 'complex', fix=True)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.win_len, self.win_type, 'complex', fix=True)

        # init
        self.gru1.apply(self.orthogonal_init_weights)
        self.linear1.apply(self.kaiming_init_weights)
        self.linear2.apply(self.kaiming_init_weights2)

    def kaiming_init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Perform Kaiming initialization
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def kaiming_init_weights2(self, module):
        if isinstance(module, nn.Linear):
            # Perform Kaiming initialization
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='sigmoid')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def orthogonal_init_weights(self, module):
        if isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    # Perform orthogonal initialization
                    nn.init.orthogonal_(param.data)

    def forward(self, mic, ref, near, erb):

        # mic = mic - (torch.mean(mic) / torch.std(mic))
        # ref = ref - (torch.mean(ref) / torch.std(ref))
        # near = near - (torch.mean(near) / torch.std(near))

        # mic = (mic - torch.min(mic, dim=1, keepdim=True)[0]) / (torch.max(mic, dim=1, keepdim=True)[0]-torch.min(mic, dim=1, keepdim=True)[0])
        # ref = (ref - torch.min(ref, dim=1, keepdim=True)[0]) / (torch.max(ref, dim=1, keepdim=True)[0] - torch.min(ref, dim=1, keepdim=True)[0])
        # near = (near - torch.min(near, dim=1, keepdim=True)[0]) / (torch.max(near, dim=1, keepdim=True)[0] - torch.min(near, dim=1, keepdim=True)[0])

        near_specs = self.cpx_stft(near)
        mic_specs = self.cpx_stft(mic)
        ref_specs = self.cpx_stft(ref)

        # split real and imag
        near_real = near_specs[:, :self.win_len // 2 + 1]
        near_imag = near_specs[:, self.win_len // 2 + 1:]

        mic_real = mic_specs[:, :self.win_len // 2 + 1]
        mic_imag = mic_specs[:, self.win_len // 2 + 1:]

        ref_real = ref_specs[:, :self.win_len // 2 + 1]
        ref_imag = ref_specs[:, self.win_len // 2 + 1:]

        # mag
        mic_mag = torch.sqrt(mic_real ** 2 + mic_imag ** 2 + 1e-9).transpose(1, 2)
        ref_mag = torch.sqrt(ref_real ** 2 + ref_imag ** 2 + 1e-9).transpose(1, 2)
        near_mag = torch.sqrt(near_real ** 2 + near_imag ** 2 + 1e-9).transpose(1, 2)

        # erb
        mic_erb = mic_mag @ erb
        ref_erb = ref_mag @ erb
        near_erb = near_mag @ erb

        # cat
        x = torch.cat([mic_erb, ref_erb], dim=2)

        # 前向传播第一层GRU
        out1, _ = self.gru1(x)

        # linear1
        out2 = self.relu(self.linear1(out1))

        # linear2
        mask = self.sigmoid(self.linear2(out2))

        # mask
        est_erb = mask * mic_erb

        mask_real = est_erb @ erb.transpose(0, 1)
        mask_imag = est_erb @ erb.transpose(0, 1)

        est_real = mask_real.transpose(1, 2) * mic_real
        est_imag = mask_imag.transpose(1, 2) * mic_imag

        # istft
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2 + 1e-9).transpose(1, 2)
        out_spec = torch.cat([est_real, est_imag], 1)
        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1) + 1e-9

        Time = near_mag.shape[1]
        Freq = erb.shape[1]
        p1 = 0.5
        loss_asym = torch.sum((F.relu(near_erb ** p1 - est_erb ** p1)) ** 2) / (Time * Freq)
        loss_mag = torch.sum(torch.abs(near_erb ** p1 - est_erb ** p1) ** 2) / (Time * Freq)
        loss = loss_mag

        # loss
        # Time = near_mag.shape[1]
        # Freq = near_mag.shape[2]
        # p1 = 0.5
        # alpha = 1
        # loss_asym = torch.sum((F.relu(near_mag ** p1 - est_mag ** p1)) ** 2) / (Time * Freq)
        # loss_mag = torch.sum(torch.abs(near_mag ** p1 - est_mag ** p1) ** 2) / (Time * Freq)
        # loss = alpha*loss_mag + (1-alpha)*loss_asym

        return out_wav, loss
        # return est_real, est_imag, near_real, near_imag, near_mag

class Little_net(nn.Module):
    def __init__(self, conf, erb_bands):
        super(Little_net, self).__init__()

        self.config = conf
        self.win_len = self.config['win_size']
        self.win_inc = self.config['hop_size']
        self.win_type = 'hann'

        # 定义第一层GRU
        self.gru1 = nn.GRU(2 * erb_bands, erb_bands, num_layers=1, batch_first=True, bias=True)

        # linear
        self.linear1 = nn.Linear(2*erb_bands, erb_bands, bias=True)
        self.linear2 = nn.Linear(erb_bands, erb_bands, bias=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # FFT
        self.cpx_stft = ConvSTFT(self.win_len, self.win_inc, self.win_len, self.win_type, 'complex', fix=True)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.win_len, self.win_type, 'complex', fix=True)

        # init
        self.gru1.apply(self.orthogonal_init_weights)
        self.linear1.apply(self.kaiming_init_weights)
        self.linear2.apply(self.kaiming_init_weights2)

    def kaiming_init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Perform Kaiming initialization
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def kaiming_init_weights2(self, module):
        if isinstance(module, nn.Linear):
            # Perform Kaiming initialization
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='sigmoid')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def orthogonal_init_weights(self, module):
        if isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    # Perform orthogonal initialization
                    nn.init.orthogonal_(param.data)

    def forward(self, mic, ref, near, erb):

        mic = mic - (torch.mean(mic) / torch.std(mic))
        ref = ref - (torch.mean(ref) / torch.std(ref))
        near = near - (torch.mean(near) / torch.std(near))

        # mic = (mic - torch.min(mic, dim=1, keepdim=True)[0]) / (torch.max(mic, dim=1, keepdim=True)[0]-torch.min(mic, dim=1, keepdim=True)[0])
        # ref = (ref - torch.min(ref, dim=1, keepdim=True)[0]) / (torch.max(ref, dim=1, keepdim=True)[0] - torch.min(ref, dim=1, keepdim=True)[0])
        # near = (near - torch.min(near, dim=1, keepdim=True)[0]) / (torch.max(near, dim=1, keepdim=True)[0] - torch.min(near, dim=1, keepdim=True)[0])

        near_specs = self.cpx_stft(near)
        mic_specs = self.cpx_stft(mic)
        ref_specs = self.cpx_stft(ref)

        # split real and imag
        near_real = near_specs[:, :self.win_len // 2 + 1]
        near_imag = near_specs[:, self.win_len // 2 + 1:]

        mic_real = mic_specs[:, :self.win_len // 2 + 1]
        mic_imag = mic_specs[:, self.win_len // 2 + 1:]

        ref_real = ref_specs[:, :self.win_len // 2 + 1]
        ref_imag = ref_specs[:, self.win_len // 2 + 1:]

        # mag
        mic_mag = torch.sqrt(mic_real ** 2 + mic_imag ** 2 + 1e-9).transpose(1, 2)
        ref_mag = torch.sqrt(ref_real ** 2 + ref_imag ** 2 + 1e-9).transpose(1, 2)
        near_mag = torch.sqrt(near_real ** 2 + near_imag ** 2 + 1e-9).transpose(1, 2)

        # erb
        mic_erb = mic_mag @ erb
        ref_erb = ref_mag @ erb
        near_erb = near_mag @ erb

        # mic-ref
        mic_ref = torch.abs(mic_erb - ref_erb)

        # cat
        x = torch.cat([mic_erb, mic_ref], dim=2)

        # 前向传播第一层GRU
        out1, _ = self.gru1(x)

        outcat = torch.cat([out1, mic_erb], dim=2)

        # linear1
        out2 = self.relu(self.linear1(outcat))

        # linear2
        mask = self.sigmoid(self.linear2(out2))

        # mask
        est_erb = mask * mic_erb

        mask_real = est_erb @ erb.transpose(0, 1)
        mask_imag = est_erb @ erb.transpose(0, 1)

        est_real = mask_real.transpose(1, 2) * mic_real
        est_imag = mask_imag.transpose(1, 2) * mic_imag

        # istft
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2 + 1e-9).transpose(1, 2)
        out_spec = torch.cat([est_real, est_imag], 1)
        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1) + 1e-9

        Time = near_mag.shape[1]
        Freq = erb.shape[1]
        p1 = 0.5
        # loss_asym = torch.sum((F.relu(near_erb ** p1 - est_erb ** p1)) ** 2) / (Time * Freq)
        loss_mag = torch.sum(torch.abs(near_erb ** p1 - est_erb ** p1) ** 2) / (Time * Freq)
        loss = loss_mag

        # loss
        # Time = near_mag.shape[1]
        # Freq = near_mag.shape[2]
        # p1 = 0.5
        # alpha = 0.5
        # loss_asym = torch.sum((F.relu(near_mag ** p1 - est_mag ** p1)) ** 2) / (Time * Freq)
        # loss_mag = torch.sum(torch.abs(near_mag ** p1 - est_mag ** p1) ** 2) / (Time * Freq)
        # loss = alpha*loss_mag + (1-alpha)*loss_asym

        return out_wav, loss
        # return est_real, est_imag, near_real, near_imag, near_mag