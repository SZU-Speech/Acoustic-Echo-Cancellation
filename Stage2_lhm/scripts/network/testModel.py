import torch
import thop
import torch.nn as nn
import torch.nn.functional as F
from thop import clever_format
from thop import profile


net_conf = {
    'conv_channels': [1, 8, 16, 32],
    'kernel_size': (1, 3),
    'stride': (1, 2),
    'padding': (0, 1),
    'dilation': 1,
    'groups': 1,
}

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
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
                            output_padding=(0, 1)
                        ),
                        # nn.BatchNorm2d(config['conv_channels'][channel_idx - 1]),
                        nn.BatchNorm2d(1),
                        nn.Tanh()
                    )
                )
        # self.gru = nn.GRU(input_size=2048, hidden_size=2048, batch_first=True, num_layers=1)

    def forward(self, noisy_dct):

        encoder_out = []
        out = noisy_dct

        # encode features
        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            encoder_out.append(out)

        # reshape
        # batch_size, channels, times, freqs = out.size()
        # out = out.transpose(1, 2).reshape(batch_size, times, channels*freqs)

        # # GRU
        # out = self.gru(out)[0]
        #
        # # reshape
        # out = out.reshape(batch_size, times, channels, freqs)
        # out = out.transpose(1, 2)

        # decoder
        for idx in range(len(self.decoder)):
            out = torch.cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)

        return out


if __name__ == "__main__":
    model = CNN(net_conf)
    input_data = torch.randn(1, 1, 1, 512)

    flops, params = profile(model, inputs=(input_data,))
    nn.GRU
    params_kb = params * 4 / (1024*1024)  # Convert bytes to KB
    print(f"Number of parameters: {params_kb:.4f} MB")
    print(f"Number of FLOPs: {flops}")
