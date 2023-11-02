import random
from datetime import datetime

import h5py
import numpy as np
import torch
import os

class WavReader(object):
    def __init__(self, in_file, mode):
        # if mode is 'train', in_file is a list of filenames;
        # if mode is 'eval', in_file is a filename
        self.mode = mode
        self.in_file = in_file
        assert self.mode in {'train', 'eval'}
        if self.mode == 'train':
            self.wav_dict = {i: wavfile for i, wavfile in enumerate(in_file)}
        else:
            reader = h5py.File(in_file, 'r')
            self.wav_dict = {i: str(i) for i in range(len(reader))}
            reader.close()
        self.wav_indices = sorted(list(self.wav_dict.keys()))

    def load(self, idx):
        if self.mode == 'train':
            son_dir = self.wav_dict[idx].replace("./", "")
            son_dir = son_dir.replace('\\', '/')
            parent_dir = os.path.abspath(os.path.dirname(os.getcwd()))
            filename = os.path.join(parent_dir, son_dir)
            reader = h5py.File(filename, 'r')
            mic = reader['mic'][:]
            ref = reader['ref'][:]
            near = reader['near'][:]
            echo = reader['echo'][:]
            reader.close()
        else:
            reader = h5py.File(self.in_file, 'r')
            reader_grp = reader[self.wav_dict[idx]]

            mic = reader_grp['mic'][:]
            ref = reader_grp['ref'][:]
            near = reader_grp['near'][:]
            echo = reader_grp['echo'][:]

            reader.close()
        return mic, ref, near, echo

    def __iter__(self):
        for idx in self.wav_indices:
            yield idx, self.load(idx)


class PerUttLoader(object):
    def __init__(self, in_file, in_norm, shuffle=True, mode='train'):
        self.shuffle = shuffle
        self.mode = mode
        self.wav_reader = WavReader(in_file, mode)
        self.in_norm = in_norm
        self.eps = np.finfo(np.float32).eps

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.wav_reader.wav_indices)

        for idx, utt in self.wav_reader:
            utt_eg = dict()
            if self.in_norm:
                utt_eg['mic'] = utt[0] / np.max(np.abs(utt[0]))
                utt_eg['ref'] = utt[1] / np.max(np.abs(utt[1]))
                utt_eg['near'] = utt[2] / np.max(np.abs(utt[2]))
                utt_eg['echo'] = utt[3] / np.max(np.abs(utt[3]))
            else:
                utt_eg['mic'] = utt[0]
                utt_eg['ref'] = utt[1]
                utt_eg['near'] = utt[2]
                utt_eg['echo'] = utt[3]
            utt_eg['n_samples'] = utt[0].shape[0]
            yield utt_eg


class SegSplitter(object):
    def __init__(self, segment_size, sample_rate, hop_size):
        self.seg_len = int(sample_rate * segment_size)
        self.hop_len = int(sample_rate * hop_size) 
        
    def __call__(self, utt_eg):
        n_samples = utt_eg['n_samples']
        segs = []
        if n_samples < self.seg_len:
            pad_size = self.seg_len - n_samples
            seg = dict()
            seg['mic'] = np.pad(utt_eg['mic'], [(0, pad_size)])
            seg['ref'] = np.pad(utt_eg['ref'], [(0, pad_size)])
            seg['near'] = np.pad(utt_eg['near'], [(0, pad_size)])
            seg['echo'] = np.pad(utt_eg['echo'], [(0, pad_size)])
            seg['n_samples'] = n_samples
            segs.append(seg)
        else:
            s_point = 0
            while True:
                if s_point + self.seg_len > n_samples:
                    break
                seg = dict()
                seg['mic'] = utt_eg['mic'][s_point:s_point+self.seg_len]
                seg['ref'] = utt_eg['ref'][s_point:s_point+self.seg_len]
                seg['near'] = utt_eg['near'][s_point:s_point + self.seg_len]
                seg['echo'] = utt_eg['echo'][s_point:s_point + self.seg_len]
                seg['n_samples'] = self.seg_len
                s_point += self.hop_len
                segs.append(seg)
        return segs


class AudioLoader(object):
    def __init__(self, 
                 in_file, 
                 sample_rate,
                 unit='seg',
                 segment_size=4.0,
                 segment_shift=1.0, 
                 batch_size=4, 
                 buffer_size=16,
                 in_norm=True,
                 mode='train'):
        self.mode = mode
        assert self.mode in {'train', 'eval'}
        self.unit = unit
        assert self.unit in {'seg', 'utt'}
        if self.mode == 'train':
            self.utt_loader = PerUttLoader(in_file, in_norm, shuffle=True, mode='train')
        else:
            self.utt_loader = PerUttLoader(in_file, in_norm, shuffle=False, mode='eval')
        if unit == 'seg':
            self.seg_splitter = SegSplitter(segment_size, sample_rate, hop_size=segment_shift)
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def make_batch(self, load_list):
        n_batches = len(load_list) // self.batch_size
        if n_batches == 0:
            return []
        else:
            batch_queue = [[] for _ in range(n_batches)]
            idx = 0
            for seg in load_list[0:n_batches*self.batch_size]:
                batch_queue[idx].append(seg)
                idx = (idx + 1) % n_batches
            if self.unit == 'utt':
                for batch in batch_queue:
                    sig_len = max([eg['mic'].shape[0] for eg in batch])
                    for i in range(len(batch)):
                        pad_size = sig_len - batch[i]['mic'].shape[0]
                        batch[i]['mic'] = np.pad(batch[i]['mic'], [(0, pad_size)])
                        batch[i]['ref'] = np.pad(batch[i]['ref'], [(0, pad_size)])
                        batch[i]['near'] = np.pad(batch[i]['near'], [(0, pad_size)])
                        batch[i]['echo'] = np.pad(batch[i]['echo'], [(0, pad_size)])

            return batch_queue

    def to_tensor(self, x):
        return torch.from_numpy(x).float()

    def batch_buffer(self):
        while True:
            try:
                utt_eg = next(self.load_iter)
                if self.unit == 'seg':
                    segs = self.seg_splitter(utt_eg)
                    self.load_list.extend(segs)
                else:
                    self.load_list.append(utt_eg)
            except StopIteration:
                self.stop_iter = True
                break
            if len(self.load_list) >= self.buffer_size:
                break
        
        batch_queue = self.make_batch(self.load_list)
        batch_list = []
        for eg_list in batch_queue:
            batch = {
                'mic': torch.stack([self.to_tensor(eg['mic']) for eg in eg_list], dim=0),
                'ref': torch.stack([self.to_tensor(eg['ref']) for eg in eg_list], dim=0),
                'near': torch.stack([self.to_tensor(eg['near']) for eg in eg_list], dim=0),
                'echo': torch.stack([self.to_tensor(eg['echo']) for eg in eg_list], dim=0),
                'n_samples': torch.tensor([eg['n_samples'] for eg in eg_list], dtype=torch.int64)
            }
            batch_list.append(batch)
        # drop used segments and keep remaining segments
        rn = len(self.load_list) % self.batch_size
        self.load_list = self.load_list[-rn:] if rn else []
        return batch_list

    def __iter__(self):
        self.load_iter = iter(self.utt_loader)
        self.stop_iter = False
        self.load_list = []
        while True:
            if self.stop_iter:
                break
            egs_buffer = self.batch_buffer()
            for egs in egs_buffer:
                yield egs

