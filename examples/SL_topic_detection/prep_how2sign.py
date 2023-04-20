#!/usr/bin/env python3

import errno
import os
import json
import h5py
import argparse
import logging
import pandas as pd
from typing import Tuple
from typing import List
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import torchvision
import torch
from torch.utils.data import Dataset
from utils import (
    gen_vocab,
    save_df_to_tsv,
    load_text,
)

log = logging.getLogger(__name__)


class How2Sign(Dataset):
    '''
    Create a Dataset for How2Sign.
    '''

    LANGUAGES = ['en'] # TODO: add 'pt'
    SPLITS = ['val', 'test', 'train']

    def __init__(
        self,
        root: str,
        lang: str,
        split: str,
        featsType: str,
    ) -> None:
        self.root = Path(root)
        self.featsType = featsType
        assert split in self.SPLITS and lang in self.LANGUAGES
        assert self.root.is_dir()

        try:
            self.h5_sign = h5py.File(self.root / f'{split}.h5', 'r')
        except:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.root / f'{split}.h5'
            )

        with h5py.File(self.root / f'{split}_filt.h5', 'w') as f:
            for key in self.h5_sign.keys():
                try:
                    f[key[:11]] = self.h5_sign[key][()]
                except:
                    pass

        self.h5_sign.close()
        self.h5_sign = h5py.File(self.root / f'{split}_filt.h5', 'r')

        if featsType == 'text':
            self.text = load_text(self.root / f'{split}.txt', list(self.h5_sign.keys()))
        elif featsType == 'spot_align' or featsType == 'mouthings':
            self.categs = pd.read_csv(self.root / f'{split}_categs.csv')

        self.data = pd.read_csv(self.root / f'{split}.csv')

        if featsType == 'text':
            self.data['TEXT'] = pd.NaT
        elif featsType == 'spot_align' or featsType == 'mouthings':
            self.data['CATEGORY_ID'] = pd.NaT
        self.data['START_FRAME'] = pd.NaT
        self.data['END_FRAME'] = pd.NaT
        for i, row in self.data.iterrows():
            if i % 20 == 0:
                print(f'iter = {i}', flush=True)
            if row['VIDEO_ID'] not in list(self.h5_sign.keys()):
                print(f'Error with keypoint {row["VIDEO_ID"]}, not found inside h5_sign', flush=True)
                self.data.drop(i, inplace=True)
            else:
                self.data.loc[i, 'START_FRAME'] = 0
                self.data.loc[i, 'END_FRAME'] = torch.Tensor(self.h5_sign[row['VIDEO_ID']]).shape[0]
                if featsType == 'text':
                    self.data.loc[i, 'TEXT'] = self.text[row['VIDEO_ID']]
                elif featsType == 'spot_align' or featsType == 'mouthings':
                    self.data.loc[i, 'CATEGORY_ID'] = self.categs[self.categs['VIDEO_ID'] == row['VIDEO_ID']]['CATEGORY_ID'].tolist()[0]
        self.data.reset_index(drop=True, inplace=True)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, str, str]:
        sent_id = self.data.loc[n, 'VIDEO_ID']
        src_signs = torch.Tensor(self.h5_sign[sent_id])
        categ = self.data.loc[n, 'CATEGORY_ID']
        if self.featsType in ['text', 'spot_align', 'mouthings']:
            text = self.data.loc[n, 'TEXT']
            return sent_id, src_signs, text, categ
        return sent_id, src_signs, categ

    def __len__(self) -> int:
        return len(self.data)

    def filter_by_length(self, min_n_frames: int, max_n_frames: int) -> None:
        lengths = self.data['END_FRAME'] - self.data['START_FRAME'] + 1
        self.data = self.data[lengths.between(min_n_frames, max_n_frames)]
        self.data.reset_index(drop=True, inplace=True)


class How2Sign_video(Dataset):
    '''
    Create a Dataset for How2Sign for video data.
    '''
    LANGUAGES = ['en'] # TODO: add 'pt'
    SPLITS = ['train', 'val', 'test']
    DATA_PATH = {
        'train': '/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/train/rgb_front/raw_videos',
        'val': '/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/val/rgb_front/raw_videos',
        'test': '/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/test/rgb_front/raw_videos',
    }
    # DATA_PATH = {
    #     'train': './',
    #     'val': './',
    #     'test': './',
    # }

    def __init__(
        self,
        root: str,
        lang: str,
        split: str,
        featsType: str,
    ) -> None:
        self.root = Path(root)
        self.featsType = featsType
        assert split in self.SPLITS and lang in self.LANGUAGES
        assert self.root.is_dir()

        self.split = split

        self.data = []

        self.videonames = self.load_video_names(os.path.join(self.root, 'subset2episode.json'))[split]
        # self.videonames = ['cKmtmtqeUkI-5-rgb_front', 'g1uA0f9I0Sg-5-rgb_front']
        self.categories = pd.read_csv(self.root / f'{split}.csv')
        self.categories = self.categories.set_index('VIDEO_ID').to_dict()['CATEGORY_ID']
        # self.categories = {'cKmtmtqeUkI': 4, 'g1uA0f9I0Sg': 2}

        for i, video_name in enumerate(self.videonames):
            cap = cv2.VideoCapture(os.path.join(self.DATA_PATH[split], video_name + '.mp4'))
            # cap = cv2.VideoCapture('/home/alvaro/Documents/ML and DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/video/cKmtmtqeUkI-5-rgb_front.mp4')
            totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("The total number of frames in this video is ", totalframecount, flush=True)

            self.data.append({
                'VIDEO_ID': video_name[:11],
                'VIDEO_NAME': video_name,
                'CATEGORY_ID': self.categories[video_name[:11]],
                'START_FRAME': 0,
                'END_FRAME': totalframecount - 1
            })

        self.data = pd.DataFrame(self.data)
        self.data = self.data.drop_duplicates(subset=['VIDEO_ID'], keep='first')
        self.data = self.data.drop_duplicates(subset=['VIDEO_NAME'], keep='first')
        self.data.reset_index(drop=True, inplace=True)

    def load_video_names(self, path: str) -> List[str]:
        with open(path, 'r') as f:
            data_features = json.load(f)
        return data_features

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, str, str]:
        sent_id = self.data.loc[n, 'VIDEO_ID']
        src_signs = torchvision.io.read_video(filename=os.path.join(self.DATA_PATH[self.split], self.data.loc[n, 'VIDEO_NAME'] + '.mp4'))
        categ = self.data.loc[n, 'CATEGORY_ID']
        return sent_id, src_signs, categ

    def __len__(self) -> int:
        return len(self.data)

    def filter_by_length(self, min_n_frames: int, max_n_frames: int) -> None:
        lengths = self.data['END_FRAME'] - self.data['START_FRAME'] + 1
        self.data = self.data[lengths.between(min_n_frames, max_n_frames)]


def process(args):
    root = Path(args.data_root).absolute()
    for split in How2Sign.SPLITS:
        print(f'Processing "{split}" split', flush=True)
        filt_csv = root / f'{split}_filt.csv'
        for lang in How2Sign.LANGUAGES:
            if args.featsType == 'video':
                dataset = How2Sign_video(root, lang, split, args.featsType)
            else:
                dataset = How2Sign(root, lang, split, args.featsType)

            print('Filtering samples by length...', flush=True)
            dataset.filter_by_length(args.min_n_frames, args.max_n_frames)
            print(f'{len(dataset)} samples remaining after filtering', flush=True)

            if split == 'train' and args.featsType in ['text', 'spot_align', 'mouthings']:
                print(f"Generating vocab for '{lang}' language")
                v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
                spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{lang}"
                with NamedTemporaryFile(mode="w") as f:
                    for i in range(len(dataset)):
                        f.write(dataset[i][2] + "\n")
                    f.seek(0)
                    gen_vocab(
                        Path(f.name),
                        root / spm_filename_prefix,
                        args.vocab_type,
                        args.vocab_size,
                        special_symbols=['_', '-']
                    )

            print('Saving dataframe...', flush=True)
            save_df_to_tsv(dataset.data, filt_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', '-d', required=True, type=str)
    parser.add_argument('--min-n-frames', default=150, type=int)
    parser.add_argument('--max-n-frames', default=5500, type=int)
    parser.add_argument('--featsType', default='keypoints', type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        type=str,
        choices=["bpe", "unigram", "char"],
    )
    parser.add_argument("--vocab-size", default=8000, type=int)

    args = parser.parse_args()

    process(args)


if __name__ == '__main__':
    main()
