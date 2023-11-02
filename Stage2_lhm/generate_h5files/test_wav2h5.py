import os
import h5py
import librosa
from glob import glob
import argparse
import numpy as np
from tqdm import tqdm


def create_h5(args):
    test_list = []
    name_list = []
    filename = 'test.ex'
    count = 0

    if not os.path.exists(os.path.join(args.h5_path, "tt")):
        os.makedirs(os.path.join(args.h5_path, "tt"))

    writer = h5py.File(os.path.join(os.path.join(args.h5_path, "tt"), filename), 'w')

    for i, nearend_speech_path in tqdm(enumerate(glob(os.path.join(args.val_path, "nearend_speech_fileid_*.wav")))):

        # get idx
        basename = os.path.basename(nearend_speech_path)
        idx = basename.split('.wav')[0].split('_')[-1]
        name_list.append(idx)

        # 读取语音
        nearend_speech, _ = librosa.load(nearend_speech_path, sr=args.sr)
        nearend_mic, _ = librosa.load(os.path.join(args.val_path, 'nearend_mic_fileid_' + idx + '.wav'), sr=args.sr)
        farend_speech, _ = librosa.load(os.path.join(args.val_path, 'farend_speech_fileid_' + idx + '.wav'), sr=args.sr)
        echo, _ = librosa.load(os.path.join(args.val_path, 'echo_fileid_' + idx + '.wav'), sr=args.sr)

        # norm
        # nearend_speech = nearend_speech / (np.max(nearend_speech) + 1e-8)
        # nearend_mic = nearend_mic / (np.max(nearend_mic) + 1e-8)
        # farend_speech = farend_speech / (np.max(farend_speech) + 1e-8)
        # echo = echo / (np.max(echo) + 1e-8)

        # 保存
        if not os.path.exists(os.path.join(args.h5_path, "tt")):
            os.makedirs(os.path.join(args.h5_path, "tt"))

        writer_grp = writer.create_group(str(count))
        writer_grp.create_dataset('nearend_speech', data=nearend_speech.astype(np.float32), shape=nearend_speech.shape, chunks=True)
        writer_grp.create_dataset('nearend_mic', data=nearend_mic.astype(np.float32), shape=nearend_mic.shape, chunks=True)
        writer_grp.create_dataset('farend_speech', data=farend_speech.astype(np.float32), shape=farend_speech.shape, chunks=True)
        writer_grp.create_dataset('echo', data=echo.astype(np.float32), shape=echo.shape, chunks=True)
        count += 1

        # if i == 50:
        #     break

    writer.close()
    test_list.append(str(os.path.join(os.path.join(args.h5_path, "tt"), filename)))
    newline = '\n'
    f = open(os.path.join(args.list_path, "tt_list.txt"), "w")
    f.write(newline.join(test_list))
    f.close()

    f = open(os.path.join(args.list_path, "filename.txt"), "w")
    f.write(newline.join(name_list))
    f.close()
    print("finish creating test h5files")


def main():
    parser = argparse.ArgumentParser(description='Configurations for turnning testing files into h5 files',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--val_path',
                        type=str,
                        default='/data/lihaoming/datasets/synthetic/test_set')

    parser.add_argument('--h5_path',
                        type=str,
                        default='/data/lihaoming/datasets/synthetic/h5')
                        # default='/data/lihaoming/datasets/synthetic/h5_small')

    parser.add_argument('--sr',
                        type=int,
                        default=16000)

    parser.add_argument('--list_path',
                        type=str,
                        default='../examples/filelists')

    args = parser.parse_args()

    if not os.path.exists(args.h5_path):
        os.makedirs(args.h5_path)

    create_h5(args)


if __name__ == '__main__':
    main()
