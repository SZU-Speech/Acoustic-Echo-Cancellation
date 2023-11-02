# some modules
import os
import h5py
import pprint
import timeit
import argparse
import numpy as np
from tqdm import tqdm

# torch modules
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler

# my modules
from configs import speech_conf, train_conf, net_conf, validate_conf, ckpt_conf, erb_conf

from network.ERB import EquivalentRectangularBandwidth, Little_net
from utils.tools import getLogger, numParams, CheckPoint, countFrames, lossLog
from test import ValidateDataset

logger = getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __getitem__(self, item):
        egs = dict()
        reader = h5py.File(self.dataset_path[item], 'r')
        egs['nearend_speech'] = np.array(reader['nearend_speech'])
        egs['nearend_mic'] = np.array(reader['nearend_mic'])
        egs['farend_speech'] = np.array(reader['farend_speech'])
        egs['echo'] = np.array(reader['echo'])
        return egs

    def __len__(self):
        return len(self.dataset_path)

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

        return {'nearend_speech': nearend_speech_tensor, 'nearend_mic': nearend_mic_tensor,
                'farend_speech': farend_speech_tensor, 'echo': echo_tensor, 'n_samples': max_len}


class Trainer(object):
    def __init__(self, args):

        # 语音参数
        self.in_norm = speech_conf['in_norm']
        self.sample_rate = speech_conf['sample_rate']
        self.win_size = speech_conf['win_size']
        self.hop_size = speech_conf['hop_size']

        # 文件
        with open(args.tr_list, 'r') as f:
            self.tr_list = [line.strip() for line in f.readlines()]
        self.tr_size = len(self.tr_list)
        self.ckpt_dir = args.ckpt_dir
        self.cv_file = args.cv_file
        self.resume_model = args.resume_model
        self.time_log = args.time_log
        self.loss_log = args.loss_log

        # 训练参数
        self.lr = train_conf['lr']
        self.lr_decay_factor = train_conf['lr_decay_factor']
        self.lr_decay_period = train_conf['lr_decay_period']
        self.clip_norm = train_conf['clip_norm']
        self.max_n_epochs = train_conf['max_n_epochs']
        self.batch_size = train_conf['batch_size']
        self.gpu_ids = train_conf['gpu_ids']
        # self.logging_period = train_conf['logging_period']
        self.logging_period = len(self.tr_list) // (self.batch_size * len(self.gpu_ids))
        # self.logging_period = 1
    def train(self):

        # 分配gpu
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            # cpu only
            self.device = torch.device('cpu')
        else:
            # gpu
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))

        # check point文件夹
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        # 训练日志
        logger = getLogger(os.path.join(self.ckpt_dir, 'train.log'), log_file=True)

        # Dataset
        train_set = TrainDataset(self.tr_list)
        validate_set = ValidateDataset(self.cv_file)

        # Dataloader
        train_loader = DataLoader(train_set, batch_size=self.batch_size * len(self.gpu_ids), shuffle=True,
                                  drop_last=True, num_workers=6, pin_memory=True, collate_fn=train_set.collate_fn)
        validate_loader = DataLoader(validate_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0,
                                     pin_memory=False)

        # 创建网络
        net = Little_net(speech_conf, erb_conf['total_erb_bands'])

        # 将网络加载到设备（gpu或cpu）上
        net = net.to(self.device)

        # 如果gpu id数目大于1则开启多线程
        if len(self.gpu_ids) > 1:
            net = DataParallel(net, device_ids=self.gpu_ids)

        # erb
        ERB = EquivalentRectangularBandwidth(erb_conf['nfreqs'], erb_conf['sample_rate'], erb_conf['total_erb_bands'],
                                                  erb_conf['low_freq'], erb_conf['max_freq'])

        self.erb = torch.tensor(ERB.filters, dtype=torch.float32).to(self.device)

        # 计算模型大小
        param_count = numParams(net)
        logger.info('Trainable parameter count: {:,d} -> {:.2f} MB\n'.format(param_count, param_count * 32 / 8 / (2 ** 20)))

        # 优化器和损失函数
        optimizer = Adam([{'params': net.parameters()}], lr=self.lr, amsgrad=False)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_period, gamma=self.lr_decay_factor)

        # 重新加载模型训练
        if self.resume_model:
            logger.info('Resuming model from {}'.format(self.resume_model))
            ckpt = CheckPoint()
            ckpt.load(self.resume_model, self.device)
            state_dict = {}
            for key in ckpt.net_state_dict:
                if len(self.gpu_ids) > 1:
                    state_dict['module.' + key] = ckpt.net_state_dict[key]
                else:
                    state_dict[key] = ckpt.net_state_dict[key]
            net.load_state_dict(state_dict)
            optimizer.load_state_dict(ckpt.optim_state_dict)
            ckpt_info = ckpt.ckpt_info
            logger.info('model info: epoch {}, iter {}, cv_loss - {:.4f}\n'.format(ckpt.ckpt_info['cur_epoch'] + 1,
                                                                                   ckpt.ckpt_info['cur_iter'] + 1,
                                                                                   ckpt.ckpt_info['cv_loss']))
        else:
            logger.info('Training from scratch...\n')

            ckpt_info = {'cur_epoch': 0, 'cur_iter': 0, 'tr_loss': None, 'cv_loss': None,
                         'best_loss': float('inf')}
            ckpt_info.update(ckpt_conf)

        # 模型训练
        start_iter = 0
        while ckpt_info['cur_epoch'] < self.max_n_epochs:
            accu_tr_loss = 0.
            accu_n_frames = 0

            # 启用 BatchNormalization 和 Dropout。
            net.train()

            with tqdm(total=len(train_loader)) as bar:

                for n_iter, egs in enumerate(train_loader):
                    n_iter += start_iter

                    # 读取语音 mic=麦克风信号 ref远端信号 near近端干净语音 echo回声信号
                    n_samples = egs['n_samples']
                    nearend_speech = egs['nearend_speech'].cuda(device=self.gpu_ids[0])
                    nearend_mic = egs['nearend_mic'].cuda(device=self.gpu_ids[0])
                    farend_speech = egs['farend_speech'].cuda(device=self.gpu_ids[0])
                    # echo = egs['echo'].cuda(device=self.gpu_ids[0])

                    # 开始计时
                    start_time = timeit.default_timer()

                    # 启用梯度计算
                    with torch.enable_grad():
                        out_wav, loss = net(nearend_mic, farend_speech, nearend_speech, self.erb)

                    # loss反向传播
                    loss.backward()

                    # 梯度裁剪
                    if self.clip_norm >= 0.0:
                        clip_grad_norm_(net.parameters(), self.clip_norm)

                    # 优化训练参数
                    optimizer.step()

                    # 计算时间
                    end_time = timeit.default_timer()
                    batch_time = end_time - start_time

                    # 平均loss
                    n_frames = countFrames(n_samples, self.win_size, self.hop_size)
                    running_loss = loss.data.item()
                    accu_tr_loss += running_loss * n_frames
                    accu_n_frames += n_frames

                    # 记录loss
                    if self.time_log:
                        with open(self.time_log, 'a+') as f:
                            print('Epoch [{}/{}], Iter [{}], tr_loss = {:.4f} / {:.4f}, batch_time (s) = {:.4f}'.format(
                                ckpt_info['cur_epoch'] + 1,
                                self.max_n_epochs, n_iter, running_loss, accu_tr_loss / accu_n_frames, batch_time), file=f)
                            f.flush()
                    else:
                        print('Epoch [{}/{}], Iter [{}], tr_loss = {:.4f} / {:.4f}, batch_time (s) = {:.4f}'.format(
                            ckpt_info['cur_epoch'] + 1,
                            self.max_n_epochs, n_iter, running_loss, accu_tr_loss / accu_n_frames, batch_time), flush=True)

                    # 记录模型
                    if (n_iter + 1) % self.logging_period == 0:
                        avg_tr_loss = accu_tr_loss / accu_n_frames
                        metrics = self.validate(net, validate_loader)
                        net.train()

                        ckpt_info['cur_iter'] = n_iter
                        ckpt_info['tr_loss'] = avg_tr_loss

                        for metric in metrics:
                            is_best = True if metrics[metric] < ckpt_info['best_' + metric] else False
                            ckpt_info['best_' + metric] = metrics[metric] if is_best else ckpt_info['best_' + metric]
                            latest_model = 'latest.pt'
                            best_model = 'best_' + metric + '.pt'
                            ckpt_info[metric] = metrics[metric]

                            if len(self.gpu_ids) > 1:
                                ckpt = CheckPoint(ckpt_info, net.module.state_dict(), optimizer.state_dict())
                            else:
                                ckpt = CheckPoint(ckpt_info, net.state_dict(), optimizer.state_dict())

                            # 保存这一iter的模型
                            logger.info('Saving checkpoint into {}'.format(os.path.join(self.ckpt_dir, latest_model)))

                            if is_best:
                                logger.info('Saving checkpoint into {}'.format(os.path.join(self.ckpt_dir, best_model)))

                            logger.info(('Epoch [{:d}/{:d}], ( tr_loss: {:.4f} | best_' + metric + ': {:.4f} )\n').format(
                                ckpt_info['cur_epoch'] + 1, self.max_n_epochs, avg_tr_loss, ckpt_info['best_' + metric]))

                            # model path
                            model_path = os.path.join(self.ckpt_dir, 'models')
                            if not os.path.isdir(model_path):
                                os.makedirs(model_path)

                            # save model
                            ckpt.save(os.path.join(model_path, latest_model), is_best, os.path.join(model_path, best_model))

                        # losslog
                        lossLog(os.path.join(self.ckpt_dir, self.loss_log), ckpt, metrics)

                        accu_tr_loss = 0.
                        accu_n_frames = 0

                        if n_iter + 1 == self.tr_size // self.batch_size:
                            start_iter = 0
                            ckpt_info['cur_iter'] = 0
                            break

                    bar.set_description(desc="Epoch %i" % ckpt_info['cur_epoch'])
                    bar.set_postfix(steps=n_iter, loss=loss.data.item())
                    bar.update(1)

            ckpt_info['cur_epoch'] += 1

            # 学习率下降
            scheduler.step()

    def validate(self, net, cv_loader):
        nums = 0
        accu_cv_loss = 0.
        accu_n_frames = 0
        buf_metrics = dict()

        # 加载网络模块
        if len(self.gpu_ids) > 1:
            net = net.module

        net.eval()
        # 开始验证
        for k, egs in enumerate(cv_loader):

            # 读取语音 mic=麦克风信号 ref远端信号 near近端干净语音 echo回声信号
            n_samples = egs['n_samples']
            nearend_speech = egs['nearend_speech'].cuda(device=self.gpu_ids[0])
            nearend_mic = egs['nearend_mic'].cuda(device=self.gpu_ids[0])
            farend_speech = egs['farend_speech'].cuda(device=self.gpu_ids[0])
            echo = egs['echo'].cuda(device=self.gpu_ids[0])

            # 不启用梯度计算
            with torch.no_grad():
                out_wav, loss = net(nearend_mic, farend_speech, nearend_speech, self.erb)

            # 求loss


            # 计算验证平均loss
            n_frames = countFrames(n_samples, self.win_size, self.hop_size)
            cv_loss = loss.data.item()
            accu_cv_loss += cv_loss * n_frames
            accu_n_frames += n_frames

            # validate size
            nums = k + 1

        # 计算验证平均loss
        avg_cv_loss = accu_cv_loss / accu_n_frames

        # average metrics
        for metric in buf_metrics:
            buf_metrics[metric] = buf_metrics[metric] / nums

        # metrics append loss
        buf_metrics.update({"loss": avg_cv_loss})
        return buf_metrics


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # parse the configurations
    parser = argparse.ArgumentParser(description='Additioal configurations for training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--tr_list',
                        type=str,
                        default=r'../examples/filelists/tr_list.txt',
                        help='Path to the list of training files')

    parser.add_argument('--cv_file',
                        type=str,
                        default='/data/lihaoming/datasets/synthetic/h5_small/tt/test.ex',
                        help='Path to the cross validation file')

    parser.add_argument('--ckpt_dir',
                        type=str,
                        required=True,
                        help='Name of the directory to dump checkpoint')

    parser.add_argument('--time_log',
                        type=str,
                        default='./exp/time.log',
                        help='Log file for timing batch processing')

    parser.add_argument('--loss_log',
                        type=str,
                        default='loss.txt',
                        help='Filename of the loss log')

    parser.add_argument('--resume_model',
                        type=str,
                        default='',
                        help='Existing model to resume training from')

    args = parser.parse_args()
    logger.info('Arguments in command:\n{}'.format(pprint.pformat(vars(args))))

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
