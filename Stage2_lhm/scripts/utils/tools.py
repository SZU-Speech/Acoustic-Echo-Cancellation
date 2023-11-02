import os
import json
import logging
import warnings
import shutil
warnings.filterwarnings("ignore")
import numpy as np
import torch


def getLogger(name,
              format_str='%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s',
              date_format='%Y-%m-%d %H:%M:%S',
              log_file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not log_file else logging.FileHandler(name)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def numParams(net):
    count = sum([int(np.prod(param.shape)) for param in net.parameters()])
    return count


def countFrames(n_samples, win_size, hop_size):
    n_overlap = win_size // hop_size
    return int((n_samples - n_overlap) // hop_size) + 1


def lossMask(shape, n_frames, device):
    loss_mask = torch.zeros(shape, dtype=torch.float32, device=device)
    for i, seq_len in enumerate(n_frames):
        loss_mask[i,0:seq_len,:] = 1.0
    return loss_mask


def lossLog(log_filename, ckpt, metrics):
    with open(log_filename, 'a') as f:
        f.write('cur_epoch={}, cur_iter={} [\n'.format(ckpt.ckpt_info['cur_epoch'] + 1, ckpt.ckpt_info['cur_iter'] + 1))
        f.write('\t')
        for metric_stype in metrics:
            f.write(metric_stype + ' = {:.4f}, '.format(metrics[metric_stype]))
        f.write('\n]\n')


def dump_json(filename, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=True)
    return


def load_json(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError('Could not find json file: {}'.format(filename))
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj


class CheckPoint(object):
    def __init__(self, ckpt_info=None, net_state_dict=None, optim_state_dict=None):
        self.ckpt_info = ckpt_info
        self.net_state_dict = net_state_dict
        self.optim_state_dict = optim_state_dict

    def save(self, filename, is_best, best_model=None):
        torch.save(self, filename)
        if is_best:
            shutil.copyfile(filename, best_model)

    def load(self, filename, device):
        if not os.path.isfile(filename):
            raise FileNotFoundError('No checkpoint found at {}'.format(filename))
        ckpt = torch.load(filename, map_location=device)
        self.ckpt_info = ckpt.ckpt_info
        self.net_state_dict = ckpt.net_state_dict
        self.optim_state_dict = ckpt.optim_state_dict
