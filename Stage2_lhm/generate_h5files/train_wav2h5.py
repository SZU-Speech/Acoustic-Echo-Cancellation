import os
import h5py
import librosa
import fnmatch
import argparse
from tqdm import tqdm
import numpy as np
from glob import glob

def create_h5(args):
    train_list = []

    for i, nearend_speech_path in tqdm(enumerate(glob(os.path.join(args.train_path, "nearend_speech_fileid_*.wav")))):

        # get idx
        basename = os.path.basename(nearend_speech_path)
        idx = basename.split('.wav')[0].split('_')[-1]

        # 读取语音
        nearend_speech, _ = librosa.load(nearend_speech_path, sr=args.sr)
        nearend_mic, _ = librosa.load(os.path.join(args.train_path, 'nearend_mic_fileid_'+idx+'.wav'), sr=args.sr)
        farend_speech, _ = librosa.load(os.path.join(args.train_path, 'farend_speech_fileid_' + idx + '.wav'), sr=args.sr)
        echo, _ = librosa.load(os.path.join(args.train_path, 'echo_fileid_' + idx + '.wav'), sr=args.sr)

        # norm
        # nearend_speech = nearend_speech / (np.max(nearend_speech)+1e-8)
        # nearend_mic = nearend_mic / (np.max(nearend_mic) + 1e-8)
        # farend_speech = farend_speech / (np.max(farend_speech) + 1e-8)
        # echo = echo / (np.max(echo) + 1e-8)

        # 保存
        if not os.path.exists(os.path.join(args.h5_path, "tr")):
            os.makedirs(os.path.join(args.h5_path, "tr"))

        tr_filename = os.path.join(args.h5_path, "tr", "tr_"+idx+".ex")
        train_list.append(str(tr_filename))

        writer = h5py.File(tr_filename, 'w')
        writer.create_dataset('nearend_speech', data=nearend_speech.astype(np.float32), shape=nearend_speech.shape, chunks=True)
        writer.create_dataset('nearend_mic', data=nearend_mic.astype(np.float32), shape=nearend_mic.shape, chunks=True)
        writer.create_dataset('farend_speech', data=farend_speech.astype(np.float32), shape=farend_speech.shape, chunks=True)
        writer.create_dataset('echo', data=echo.astype(np.float32), shape=echo.shape, chunks=True)

        writer.close()
        # if i == 500:
        #     break

    newline = '\n'
    f = open(os.path.join(args.list_path, "tr_list.txt"), "w")
    f.write(newline.join(train_list))
    f.close()
    print("finish creating trainning h5files")


def main():
    parser = argparse.ArgumentParser(description='Configurations for turnning training files into h5 files',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_path',
                        type=str,
                        default='/data/lihaoming/datasets/synthetic/train_set')

    parser.add_argument('--h5_path',
                        type=str,
                        default='/data/lihaoming/datasets/synthetic/h5')

    parser.add_argument('--list_path',
                        type=str,
                        default='../examples/filelists')

    parser.add_argument('--sr',
                        type=int,
                        default=16000)


    args = parser.parse_args()

    if not os.path.exists(args.h5_path):
        os.makedirs(args.h5_path)

    create_h5(args)


if __name__ == '__main__':

    main()
