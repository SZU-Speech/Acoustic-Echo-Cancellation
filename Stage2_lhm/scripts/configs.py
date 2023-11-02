speech_conf = {
    'in_norm': True,
    'sample_rate': 16000,
    'win_len': 0.032,
    'hop_len': 0.016,
    'win_size': 512,
    'hop_size': 256
}

train_conf = {
    'logging_period': 1,
    'lr': 0.00001,
    'lr_decay_factor': 0.5,
    'lr_decay_period': 5,
    'clip_norm': -1,
    'max_n_epochs': 50,
    'batch_size': 16,
    'gpu_ids': [0]
}

erb_conf = {
    'nfreqs': 257,
    'sample_rate': 16000,
    'total_erb_bands': 32,
    'low_freq': 0,
    'max_freq': 8000
}

net_conf = {
    'win_size': 512,
    'hop_size': 256,
    'samplerates': 16000,
    'win_type': 'hann',
    'hidden_dim': 4,
    'rnn_layers': 2,
    'rnn_units': 128,
    'use_clstm': True,
    'use_cbn': True,
    'masking_mode': 'E',
    'conv_channels': [4, 16, 32, 64, 128, 256, 512],
    'kernel_size': (5, 1),
    'stride': (2, 1),
    'padding': (2, 0),
    'dilation': 1,
    'groups': 1,
}


validate_conf = {
    'metrics_type': ['stoi', 'sisdr'],
    'metrcis_length': None
}

ckpt_conf = {}
for metric_type in validate_conf['metrics_type']:
    ckpt_conf['cv_' + metric_type] = None
    ckpt_conf['best_'+metric_type] = float('inf')


