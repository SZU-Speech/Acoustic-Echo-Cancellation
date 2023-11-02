import os
import h5py
import librosa
from glob import glob
import argparse
import numpy as np
from tqdm import tqdm


def create_h5(args):
    farend_dir = os.path.join(args.val_path, "farend_speech")
    nearend_speech_dir = os.path.join(args.val_path, "nearend_speech")
    nearend_mic_dir = os.path.join(args.val_path, "nearend_mic")
    echo_dir = os.path.join(args.val_path, "echo")
    test_list = []
    name_list = []
    filename = 'test2.ex'
    count = 0

    if not os.path.exists(os.path.join(args.h5_path, "tt")):
        os.makedirs(os.path.join(args.h5_path, "tt"))

    writer = h5py.File(os.path.join(os.path.join(args.h5_path, "tt"), filename), 'w')
    for i, mic_path in tqdm(enumerate(glob(os.path.join(nearend_mic_dir, "*.wav")))):

        # get idx
        basename = os.path.basename(mic_path)
        name_list.append(basename)
        basename = basename.split('_')[-1]
        idx = basename.split('.wav')[0]

        # 读取语音
        mic, _ = librosa.load(mic_path, sr=args.sr)
        ref, _ = librosa.load(os.path.join(farend_dir, "farend_speech_fileid_"+idx+".wav"), sr=args.sr)
        near, _ = librosa.load(os.path.join(nearend_speech_dir, "nearend_speech_fileid_"+idx+".wav"), sr=args.sr)
        echo, _ = librosa.load(os.path.join(echo_dir, "echo_fileid_"+idx+".wav"), sr=args.sr)

        # 保存
        if not os.path.exists(os.path.join(args.h5_path, "tt")):
            os.makedirs(os.path.join(args.h5_path, "tt"))

        writer_grp = writer.create_group(str(count))
        writer_grp.create_dataset('mic', data=mic.astype(np.float32), shape=mic.shape, chunks=True)
        writer_grp.create_dataset('ref', data=ref.astype(np.float32), shape=ref.shape, chunks=True)
        writer_grp.create_dataset('near', data=near.astype(np.float32), shape=near.shape, chunks=True)
        writer_grp.create_dataset('echo', data=echo.astype(np.float32), shape=near.shape, chunks=True)
        count += 1

    writer.close()
    test_list.append(str(os.path.join(os.path.join(args.h5_path, "tt"), filename)))
    newline = '\n'
    f = open(os.path.join(args.list_path, "tt_list2.txt"), "w")
    f.write(newline.join(test_list))
    f.close()

    f = open(os.path.join(args.list_path, "filename.txt"), "w")
    f.write(newline.join(name_list))
    f.close()
    print("finish creating test h5files")


def main():
    parser = argparse.ArgumentParser(description='Configurations for turnning training files into h5 files',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--val_path',
                        type=str,
                        default='/data/lihaoming/gen_data/data/test_sets',
                        help='training_data inluding mic ref nearend')

    parser.add_argument('--h5_path',
                        type=str,
                        default='/data/lihaoming/gen_data/data/h5',
                        help='folder for h5 files')

    parser.add_argument('--sr',
                        type=int,
                        default=16000,
                        help='folder for h5 files')

    parser.add_argument('--list_path',
                        type=str,
                        default='../examples/filelists',
                        help='folder for train list')

    args = parser.parse_args()

    if not os.path.exists(args.h5_path):
        os.makedirs(args.h5_path)

    create_h5(args)


if __name__ == '__main__':
    main()
