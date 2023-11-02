import argparse
import pprint
import h5py
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.tools import getLogger, numParams, CheckPoint
from configs import speech_conf, train_conf, net_conf, validate_conf, ckpt_conf, erb_conf
from network.ERB import EquivalentRectangularBandwidth, Little_net
logger = getLogger(__name__)


class ValidateDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.reader = h5py.File(self.dataset_path, 'r')
        self.wav_dict = {i: str(i) for i in range(len(self.reader))}

    def __getitem__(self, item):
        egs = dict()
        reader_grp = self.reader[self.wav_dict[item]]
        egs['nearend_speech'] = np.array(reader_grp['nearend_speech'])
        egs['nearend_mic'] = np.array(reader_grp['nearend_mic'])
        egs['farend_speech'] = np.array(reader_grp['farend_speech'])
        egs['echo'] = np.array(reader_grp['echo'])
        egs['n_samples'] = len(egs['nearend_speech'])
        return egs

    def __len__(self):
        return len(self.wav_dict)

    @staticmethod
    def collate_fn(data_list):
        nearend_speech_list = []
        nearend_mic_list = []
        farend_speech_list = []
        echo_list = []

        max_len = 0
        for minibatch in data_list:
            length = len(minibatch['nearend_speech'])
            if length > max_len:
                max_len = length
        for idx, minibatch in enumerate(data_list):
            minibatch['nearend_speech'] = np.pad(minibatch['nearend_speech'], (0, max_len - len(minibatch['nearend_speech'])), 'constant')
            minibatch['nearend_mic'] = np.pad(minibatch['nearend_mic'], (0, max_len - len(minibatch['nearend_mic'])), 'constant')
            minibatch['farend_speech'] = np.pad(minibatch['farend_speech'], (0, max_len - len(minibatch['farend_speech'])), 'constant')
            minibatch['echo'] = np.pad(minibatch['echo'], (0, max_len - len(minibatch['echo'])), 'constant')

            nearend_speech_list.append(minibatch['nearend_speech'])
            nearend_mic_list.append(minibatch['nearend_mic'])
            farend_speech_list.append(minibatch['farend_speech'])
            echo_list.append(minibatch['echo'])

        nearend_speech_tensor = torch.tensor(nearend_speech_list, dtype=torch.float32)
        nearend_mic_tensor = torch.tensor(nearend_mic_list, dtype=torch.float32)
        farend_speech_tensor = torch.tensor(farend_speech_list, dtype=torch.float32)
        echo_tensor = torch.tensor(echo_list, dtype=torch.float32)

        return {'nearend_speech': nearend_speech_tensor, 'nearend_mic': nearend_mic_tensor, 'farend_speech': farend_speech_tensor,
                'echo': echo_tensor, 'n_samples': max_len}


class Tester(object):
    def __init__(self, args):

        # 语音参数
        self.in_norm = speech_conf['in_norm']
        self.sample_rate = speech_conf['sample_rate']
        self.win_size = speech_conf['win_size']
        self.hop_size = speech_conf['hop_size']
        self.gpu_ids = train_conf['gpu_ids']

        # 分配gpu
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            # cpu only
            self.device = torch.device('cpu')
        else:
            # gpu
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))

        # 文件路径
        self.model_file = args.model_file
        self.ckpt_dir = args.ckpt_dir
        self.est_path = args.est_path
        self.fimename_list = args.filename_list
        with open(args.tt_list, 'r') as f:
            self.tt_list = [line.strip() for line in f.readlines()]

    def test(self):

        # 创建检查点
        logger = getLogger(os.path.join(self.ckpt_dir, 'test.log'), log_file=True)

        # 创建网络
        net = Little_net(speech_conf, erb_conf['total_erb_bands'])

        # 将网络加载到设备（gpu或cpu）上
        net = net.to(self.device)

        # erb
        ERB = EquivalentRectangularBandwidth(erb_conf['nfreqs'], erb_conf['sample_rate'], erb_conf['total_erb_bands'],
                                             erb_conf['low_freq'], erb_conf['max_freq'])

        self.erb = torch.tensor(ERB.filters, dtype=torch.float32).to(self.device)

        # 计算模型大小
        logger.info('backbone summary:\n{}'.format(net))

        # 计算模型大小
        param_count = numParams(net)
        logger.info('Trainable parameter count: {:,d} -> {:.2f} MB\n'.format(param_count, param_count * 32 / 8 / (2 ** 20)))

        # 加载训练保存好的模型
        logger.info('Loading model from {}'.format(self.model_file))
        ckpt = CheckPoint()
        ckpt.load(self.model_file, self.device)
        net.load_state_dict(ckpt.net_state_dict)
        # logger.info('model info: epoch {}, iter {}, cv_loss - {:.4f}\n'.format(ckpt.ckpt_info['cur_epoch'] + 1,
        #                                                                        ckpt.ckpt_info['cur_iter'] + 1,
        #                                                                        ckpt.ckpt_info['cv_loss']))

        # filename
        with open(self.fimename_list) as f:
            flist = f.readlines()

        # 停用BatchNormalization 和 Dropout。
        net.eval()

        # 开始测试
        for i in range(len(self.tt_list)):
            tt_set = ValidateDataset(self.tt_list[0])
            tt_loader = DataLoader(tt_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

            # logger.info('[{}/{}] Estimating on {}'.format(i + 1, len(self.tt_list), self.tt_list[i]))
            est_subdir = os.path.join(self.est_path, self.tt_list[i].split('/')[-1].replace('.ex', ''))

            if not os.path.isdir(est_subdir):
                os.makedirs(est_subdir)

            net.eval()

            for k, egs in tqdm(enumerate(tt_loader)):
                nearend_speech = egs['nearend_speech'].cuda(device=self.gpu_ids[0])
                nearend_mic = egs['nearend_mic'].cuda(device=self.gpu_ids[0])
                farend_speech = egs['farend_speech'].cuda(device=self.gpu_ids[0])
                echo = egs['echo'].cuda(device=self.gpu_ids[0])

                # 不启用梯度计算
                with torch.no_grad():
                    out_wav, _ = net(nearend_mic, farend_speech, nearend_speech, self.erb)

                out_wav = out_wav[0].cpu().numpy()
                nearend_speech = nearend_speech[0].cpu().numpy()
                nearend_mic = nearend_mic[0].cpu().numpy()
                farend_speech = farend_speech[0].cpu().numpy()
                echo = echo[0].cpu().numpy()

                sf.write(os.path.join(est_subdir, str(k)+'_near_est.wav'), out_wav, self.sample_rate)
                sf.write(os.path.join(est_subdir, str(k)+'_near.wav'), nearend_speech, self.sample_rate)
                sf.write(os.path.join(est_subdir, str(k) + '_far.wav'), farend_speech, self.sample_rate)
                sf.write(os.path.join(est_subdir, str(k) + '_mic.wav'), nearend_mic, self.sample_rate)
                sf.write(os.path.join(est_subdir, str(k) + '_echo.wav'), echo, self.sample_rate)


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # parse the configuarations
    parser = argparse.ArgumentParser(description='Additioal configurations for testing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tt_list',
                        type=str,
                        required=True,
                        default='../examples/filelists/tt_list.txt',
                        help='Path to the list of testing files')

    parser.add_argument('--filename_list',
                        type=str,
                        required=True,
                        default='../examples/filelists/filename.txt')

    parser.add_argument('--ckpt_dir',
                        type=str,
                        required=True,
                        default='exp')

    parser.add_argument('--model_file',
                        type=str,
                        default='./exp/models/best_loss.pt',
                        help='Path to the model file')

    parser.add_argument('--est_path',
                        type=str,
                        default='/data/lihaoming/datasets/synthetic/estimate',
                        help='Path to dump estimates')

    args = parser.parse_args()
    logger.info('Arguments in command:\n{}'.format(pprint.pformat(vars(args))))

    tester = Tester(args)
    tester.test()


if __name__ == '__main__':
    main()
