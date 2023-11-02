import os
import shutil
import timeit
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler


from utils.utils import getLogger, numParams, lossLog
from utils.data_utils import AudioLoader
from utils.fullsubnet import FullSubNet
from utils.stft import STFT
from utils.criteria import si_snr

from python_speech_features import get_filterbanks


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


def padding(wav_change, length):
    if length > wav_change.shape[1]:
        pad_len = length - wav_change.shape[1]
        wav_change = F.pad(wav_change, (0, pad_len), )
        F.pad(input=wav_change, pad=(0, pad_len), mode='constant', value=0)
    return wav_change


class Model(object):
    def __init__(self):

        self.in_norm = True
        self.sample_rate = 16000
        self.win_size = 0.020
        self.hop_size = 0.010


    def train(self, args):
        with open(args.tr_list, 'r') as f:
            self.tr_list = [line.strip() for line in f.readlines()]
        self.tr_size = len(self.tr_list)
        self.cv_file = args.cv_file
        self.ckpt_dir = args.ckpt_dir
        self.logging_period = args.logging_period
        self.resume_model = args.resume_model
        self.time_log = args.time_log
        self.lr = args.lr
        self.lr_decay_factor = args.lr_decay_factor
        self.lr_decay_period = args.lr_decay_period
        self.clip_norm = args.clip_norm
        self.max_n_epochs = args.max_n_epochs
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.loss_log = args.loss_log
        self.unit = args.unit
        self.segment_size = args.segment_size
        self.segment_shift = args.segment_shift
        # self.gpu_ids = args.gpu_ids
        self.gpu_ids = [0]

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

        # 训练和验证的dataloader
        tr_loader = AudioLoader(self.tr_list, self.sample_rate, self.unit,
                                self.segment_size, self.segment_shift,
                                self.batch_size*len(self.gpu_ids), self.buffer_size*len(self.gpu_ids),
                                self.in_norm, mode='train')
        cv_loader = AudioLoader(self.cv_file, self.sample_rate, unit='utt',
                                segment_size=10.0, segment_shift=0.0,
                                batch_size=1, buffer_size=1,
                                in_norm=self.in_norm, mode='eval')

        # 创建网络
        net = FullSubNet(batch_size=self.batch_size, win_len=320, hop_len=160)

        # 网络构造信息
        logger.info('Backebone summary:\n{}'.format(net))

        # 将网络加载到设备（gpu或cpu）上
        net = net.to(self.device)
        self.stft = STFT(win_size=320, hop_size=160, feature_type='complex').to(self.device)
        self.fbank = torch.from_numpy(get_filterbanks(nfilt=21, nfft=320, samplerate=16000, lowfreq=20, highfreq=8000)).float().to(self.device)

        # 如果gpu id数目大于1则开启多线程
        if len(self.gpu_ids) > 1:
            net = DataParallel(net, device_ids=self.gpu_ids)

        # 计算模型大小
        param_count = numParams(net)
        sizenet = param_count * 32 / 8 / (2 ** 20)
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
            ckpt_info = {'cur_epoch': 0,
                         'cur_iter': 0,
                         'tr_loss': None,
                         'cv_loss': None,
                         'best_loss': float('inf')}

        # 模型训练
        start_iter = 0
        while ckpt_info['cur_epoch'] < self.max_n_epochs:
            accu_tr_loss = 0.
            accu_n_frames = 0

            # 启用 BatchNormalization 和 Dropout。
            net.train()
            for n_iter, egs in tqdm(enumerate(tr_loader)):

                n_iter += start_iter

                # 读取语音 mic=麦克风信号 ref远端信号 near近端干净语音 echo回声信号
                mic = egs['mic']
                ref = egs['ref']
                near = egs['near']
                # echo = egs['echo']

                # 将语音加载进设备（gpu或cpu）
                mic = mic.cuda(device=self.gpu_ids[0])
                ref = ref.cuda(device=self.gpu_ids[0])
                near = near.cuda(device=self.gpu_ids[0])
                # echo = echo.cuda(device=self.gpu_ids[0])

                # near spec
                near_real, near_imag = self.stft.stft(near)

                # 开始计时
                start_time = timeit.default_timer()

                # 梯度清零
                optimizer.zero_grad()

                # 启用梯度计算
                with torch.enable_grad():
                    output_real, output_imag = net(mic, ref, self.fbank)

                # loss
                loss_real = F.mse_loss(output_real.squeeze(1), near_real, reduction='mean')
                loss_imag = F.mse_loss(output_imag.squeeze(1), near_imag, reduction='mean')
                loss = loss_real + loss_imag

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

                # 计算loss
                n_frames = self.batch_size * near_real.shape[2]
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
                    avg_cv_loss = self.validate(net, cv_loader)
                    net.train()

                    ckpt_info['cur_iter'] = n_iter
                    is_best = True if avg_cv_loss < ckpt_info['best_loss'] else False
                    ckpt_info['best_loss'] = avg_cv_loss if is_best else ckpt_info['best_loss']
                    latest_model = 'latest.pt'
                    best_model = 'best.pt'
                    ckpt_info['tr_loss'] = avg_tr_loss
                    ckpt_info['cv_loss'] = avg_cv_loss
                    if len(self.gpu_ids) > 1:
                        ckpt = CheckPoint(ckpt_info, net.module.state_dict(), optimizer.state_dict())
                    else:
                        ckpt = CheckPoint(ckpt_info, net.state_dict(), optimizer.state_dict())
                    logger.info('Saving checkpoint into {}'.format(os.path.join(self.ckpt_dir, latest_model)))
                    if is_best:
                        logger.info('Saving checkpoint into {}'.format(os.path.join(self.ckpt_dir, best_model)))
                    logger.info(
                        'Epoch [{}/{}], ( tr_loss: {:.4f} | cv_loss: {:.4f} )\n'.format(ckpt_info['cur_epoch'] + 1,
                                                                                        self.max_n_epochs, avg_tr_loss,
                                                                                        avg_cv_loss))

                    model_path = os.path.join(self.ckpt_dir, 'models')
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)

                    ckpt.save(os.path.join(model_path, latest_model),
                              is_best,
                              os.path.join(model_path, best_model))
                    #
                    lossLog(os.path.join(self.ckpt_dir, self.loss_log), ckpt, self.logging_period)

                    accu_tr_loss = 0.
                    accu_n_frames = 0

                    if n_iter + 1 == self.tr_size // self.batch_size:
                        start_iter = 0
                        ckpt_info['cur_iter'] = 0
                        break

            ckpt_info['cur_epoch'] += 1

            # 学习率下降
            scheduler.step()
        return

    def validate(self, net, cv_loader):

        accu_cv_loss = 0.
        accu_n_frames = 0

        # 加载网络模块
        if len(self.gpu_ids) > 1:
            net = net.module

        net.eval()
        # 开始验证
        for k, egs in enumerate(cv_loader):

            # 读取语音 mic=麦克风信号 ref远端信号 near近端干净语音 echo回声信号
            mic = egs['mic']
            ref = egs['ref']
            near = egs['near']
            # echo = egs['echo']

            # 将语音加载进设备（gpu或cpu）
            mic = mic.cuda(device=self.gpu_ids[0])
            ref = ref.cuda(device=self.gpu_ids[0])
            near = near.cuda(device=self.gpu_ids[0])
            # echo = echo.cuda(device=self.gpu_ids[0])

            # near spec
            near_real, near_imag = self.stft.stft(near)

            # 不启用梯度计算
            with torch.no_grad():
                output_real, output_imag = net(mic, ref, self.fbank)

            # loss
            loss_real = F.mse_loss(output_real.squeeze(1), near_real, reduction='mean')
            loss_imag = F.mse_loss(output_imag.squeeze(1), near_imag, reduction='mean')
            loss = loss_real + loss_imag

            # 计算验证平均loss
            n_frames = self.batch_size * near_real.shape[2]
            cv_loss = loss.data.item()
            accu_cv_loss += cv_loss * n_frames
            accu_n_frames += n_frames
        avg_cv_loss = accu_cv_loss / accu_n_frames
        return avg_cv_loss

    def test(self, args):
        with open(args.tt_list, 'r') as f:
            self.tt_list = [line.strip() for line in f.readlines()]
        self.model_file = args.model_file
        self.ckpt_dir = args.ckpt_dir
        self.est_path = args.est_path
        self.write_ideal = args.write_ideal
        self.gpu_ids = tuple(map(int, args.gpu_ids.split(',')))

        # 分配gpu
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            # cpu only
            self.device = torch.device('cpu')
        else:
            # gpu
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))

        # 创建检查点
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        logger = getLogger(os.path.join(self.ckpt_dir, 'test.log'), log_file=True)

        # 创建网络
        net = FullSubNet()

        # 将网络加载进设备
        net = net.to(self.device)
        self.stft = STFT(320, 160, feature_type='real').to(self.device)

        # 计算模型大小
        logger.info('backbone summary:\n{}'.format(net))

        # 计算模型大小
        param_count = numParams(net)
        logger.info(
            'Trainable parameter count: {:,d} -> {:.2f} MB\n'.format(param_count, param_count * 32 / 8 / (2 ** 20)))

        # 加载训练保存好的模型
        logger.info('Loading model from {}'.format(self.model_file))
        ckpt = CheckPoint()
        ckpt.load(self.model_file, self.device)
        net.load_state_dict(ckpt.net_state_dict)
        logger.info('model info: epoch {}, iter {}, cv_loss - {:.4f}\n'.format(ckpt.ckpt_info['cur_epoch'] + 1,
                                                                               ckpt.ckpt_info['cur_iter'] + 1,
                                                                               ckpt.ckpt_info['cv_loss']))

        # 停用BatchNormalization 和 Dropout。
        net.eval()

        with open(args.filename_list) as f:
            flist = f.readlines()

        # 开始测试
        for i in range(len(self.tt_list)):

            # create a data loader for testing
            tt_loader = AudioLoader(self.tt_list[i], self.sample_rate, unit='utt',
                                    segment_size=None, segment_shift=None,
                                    batch_size=1, buffer_size=0,
                                    in_norm=self.in_norm, mode='eval')

            logger.info('[{}/{}] Estimating on {}'.format(i + 1, len(self.tt_list), self.tt_list[i]))
            est_subdir = os.path.join(self.est_path, self.tt_list[i].split('/')[-1].replace('.ex', ''))

            if not os.path.isdir(est_subdir):
                os.makedirs(est_subdir)

            net.eval()
            for k, egs in tqdm(enumerate(tt_loader)):
                # 读取语音 mic=麦克风信号 ref远端信号 near近端干净语音（label）n_samples语音长度（点数）
                mic = egs['mic']
                ref = egs['ref']
                near = egs['near']
                echo = egs['echo']

                # 将语音加载进设备（gpu或cpu）
                mic = mic.cuda(device=self.gpu_ids[0])
                mic = padding(mic, 160000)
                ref = ref.cuda(device=self.gpu_ids[0])
                ref = padding(ref, 160000)
                near = near.cuda(device=self.gpu_ids[0])
                near = padding(near, 160000)
                echo = echo.cuda(device=self.gpu_ids[0])
                echo = padding(echo, 160000)

                # stft
                mic_mag, mic_pha = self.stft.stft(mic)
                ref_mag, ref_pha = self.stft.stft(ref)
                near_mag, near_pha = self.stft.stft(near)
                echo_mag, echo_pha = self.stft.stft(echo)

                # Group Delay
                mic_gd = torch.diff(mic_pha, dim=2)
                mic_gd = F.pad(mic_gd, [1, 0, 0, 0, 0, 0], "constant", 0)
                ref_gd = torch.diff(ref_pha, dim=2)
                ref_gd = F.pad(ref_gd, [1, 0, 0, 0, 0, 0], "constant", 0)

                # 不启用梯度计算
                with torch.no_grad():
                    mask_near, mask_echo = net(torch.stack([mic_mag, ref_mag], dim=1))

                # pha
                pha = (1 + mask_near ** 2 - mask_echo ** 2) / (2 * mask_near + 1e-9)

                # 近端语音
                est_mag = mask_near * mic_mag
                est_pha = mic_pha
                est_real = est_mag * torch.cos(est_pha)
                est_imag = est_mag * torch.sin(est_pha)
                near_est = self.stft.istft(torch.stack([est_real, est_imag], dim=1).transpose(2, 3))
                near_est = near_est[0].cpu().numpy()

                # 回声语音
                est_mag = mask_echo * mic_mag
                est_pha = mic_pha
                est_real = est_mag * torch.cos(est_pha)
                est_imag = est_mag * torch.sin(est_pha)
                echo_est = self.stft.istft(torch.stack([est_real, est_imag], dim=1).transpose(2, 3))
                echo_est = echo_est[0].cpu().numpy()

                mic = mic[0].cpu().numpy()
                near = near[0].cpu().numpy()
                ref = ref[0].cpu().numpy()
                echo = echo[0].cpu().numpy()

                fname = flist[k].split('.wav')[-2]
                fname = fname.split('_')[-1]

                sf.write(os.path.join(est_subdir, fname + '_mic.wav'), mic, self.sample_rate)
                sf.write(os.path.join(est_subdir, fname + '_near.wav'), near, self.sample_rate)
                sf.write(os.path.join(est_subdir, fname + '_ref.wav'), ref, self.sample_rate)
                sf.write(os.path.join(est_subdir, fname + '_echo.wav'), echo, self.sample_rate)
                sf.write(os.path.join(est_subdir, fname + '_near_est.wav'), near_est, self.sample_rate)
                sf.write(os.path.join(est_subdir, fname + '_echo_est.wav'), echo_est, self.sample_rate)
        return
