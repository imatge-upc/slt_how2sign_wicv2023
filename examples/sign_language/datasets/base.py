from typing import Tuple
from pathlib import Path

import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset

from pose_format.pose import Pose


class SignLanguageDataset(Dataset):
    MANIFEST_COLUMNS = ['id', 'signs_file', 'signs_offset', 'signs_length',
                        'signs_type', 'signs_lang', 'translation',
                        'translation_lang', 'glosses', 'topic', 'signer_id']

    def __init__(self, base_dir: Path, split: str, signs_type: str,
                 signs_lang: str, translation_lang: str) -> None:
        base_dir = base_dir.expanduser().resolve()

        assert base_dir.is_dir()
        assert split in self.SPLITS
        assert signs_type in self.SIGNS_TYPES
        assert signs_lang in self.SIGNS_LANGS
        assert translation_lang in self.TRANSLATION_LANGS

        self.base_dir = base_dir
        self.split = split
        self.signs_type = signs_type
        self.signs_lang = signs_lang
        self.translation_lang = translation_lang

    def __getitem__(self, n: int) -> Tuple[Tensor, str, str]:
        row = self.data.loc[n, :]

        # This check is needed for some text-to-sign test sets,
        # where target is not provided
        if not pd.isna(row['signs_file']):
            signs_file = row.pop('signs_file')
            offset = row.pop('signs_offset')
            length = row.pop('signs_length')

            with open(signs_file, 'rb') as f:
                p = Pose.read(f.read())
            p.body = p.body.select_frames(list(range(offset, offset+length)))
            row['signs'] = p.torch()

        sample = row.to_dict()
        return {k: v for k, v in sample.items() if not pd.isna(v)}

    def __len__(self) -> int:
        return len(self.data)

    def filter_by_length(self, min_n_frames: int, max_n_frames: int) -> None:
        pre_len = len(self.data)
        self.data = self.data[self.data['signs_length'].between(min_n_frames, max_n_frames)]
        post_len = len(self.data)
        return pre_len - post_len
