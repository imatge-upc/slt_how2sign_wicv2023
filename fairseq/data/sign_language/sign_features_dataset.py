import sys
import logging
from enum import Enum
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import mediapipe as mp
from pose_format import Pose

from fairseq.data import FairseqDataset

logger = logging.getLogger(__name__)

mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [
    str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))
]
POSE_RM = ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
           'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
POSE_POINTS = [kp.name for kp in mp_holistic.PoseLandmark if kp.name not in POSE_RM]


# TODO: Adapt implementation for i3d and CNN2d
class SignFeatsType(str, Enum):
    mediapipe = "mediapipe"
    i3d = "i3d"
    CNN2d = "CNN2d"


class NormType(str, Enum):
    body = "body"
    kp_wise = "kp_wise"
    global_xyz = "global_xyz"


class SignFeatsDataset(FairseqDataset):
    def __init__(
        self,
        ids: List[str],
        feats_files: List[Union[Path, str]],
        offsets: List[int],
        sizes: List[int],
        feats_type: SignFeatsType,
        normalization: NormType = NormType.body,
        data_augmentation: bool = False,
        min_sample_size: int = 0,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
    ):
        super().__init__()
        assert len(ids) == len(feats_files) == len(offsets) == len(sizes)

        self.ids = ids
        self.feats_files = feats_files
        self.offsets = offsets
        self.sizes = sizes
        self.feats_type = feats_type
        self.normalization = normalization
        self.data_augmentation = data_augmentation
        self.min_sample_size = min_sample_size
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.shuffle = shuffle
        self.skipped_ids = []

    def filter_by_length(self, min_sample_size, max_sample_size):
        for _id, size in zip(self.ids[:], self.sizes[:]):
            if size < self.min_sample_size or size > self.max_sample_size:
                self.feats_files.pop(self.ids.index(_id))
                self.offsets.pop(self.ids.index(_id))
                self.sizes.pop(self.ids.index(_id))
                self.ids.remove(_id)
                self.skipped_ids.append(_id)
        logger.info(
            f"Filtered {len(self.skipped_ids)} sentences, that were too short or too long."
        )

    @classmethod
    def from_manifest_file(cls, manifest_file: Union[str, Path], **kwargs):
        ids = []
        feats_files = []
        offsets = []
        sizes = []
        manifest = pd.read_csv(manifest_file, sep="\t")
        for _, row in manifest.iterrows():
            ids.append(row['id'])
            feats_files.append(row['signs_file'])
            offsets.append(int(row['signs_offset']))
            sizes.append(int(row['signs_length']))
        logger.info(f"loaded {len(ids)} samples")
        
        # FIXME: This is too simplistic, we will improve it in the future
        feats_type = row['signs_type']

        return cls(ids, feats_files=feats_files, offsets=offsets, sizes=sizes,
                   feats_type=feats_type, **kwargs)

    def __getitem__(self, index):
        _id = self.ids[index]
        feats_file = self.feats_files[index]
        offset = self.offsets[index]
        length = self.sizes[index]

        with open(feats_file, "rb") as f:
            pose = Pose.read(f.read())

        frames_list = list(range(offset, offset+length))

        # Fix to bypass some examples that are wrong
        frames_list = [fr for fr in frames_list if fr < pose.body.data.shape[0]]

        pose.body = pose.body.select_frames(frames_list)

        pose = self.postprocess(pose)

        return {"id": index, "vid_id": _id, "source": pose}

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, pose):
        pose = pose.get_components(
            ["FACE_LANDMARKS", "POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
            {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS, "POSE_LANDMARKS": POSE_POINTS}
        )

        if self.normalization == NormType.body:
            normalize_info = pose.header.normalization_info(
                p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
            )
            pose.normalize(normalize_info)
        elif self.normalization == NormType.kp_wise:
            mean, std = pose.normalize_distribution(axis=(0, 1))
        elif self.normalization == NormType.global_xyz:
            mean, std = pose.normalize_distribution(axis=(0, 1, 2))
        else:
            raise ValueError(f"Unknown normalization '{self.normalization}'")

        if self.data_augmentation:
            pose = pose.augment2d()

        return pose.torch()

    def collater(self, samples):
        max_length = max([s['source'].body.data.shape[0] for s in samples])
        
        ids = []
        padding_masks = []
        collated_sources = []
        for sample in samples:
            pose = sample['source']

            if pose.body.data.shape[1] > 1:
                logger.warning(f"More than one person in frame, keeping just the first one")
            pose.body.data = pose.body.data[:, 0]

            padding_mask = (~pose.body.data.mask).sum((1,2)) > 0
            if padding_mask.all():
                continue

            diff_length = max_length - len(padding_mask)
            ids.append(sample['id'])
            padding_masks.append(
                F.pad(padding_mask, (0, diff_length), value=True)
            )
            collated_sources.append(
                F.pad(pose.body.data.data, (0, 0, 0, 0, 0, diff_length), value=0)
            )

        if len(collated_sources) == 0:
            return {}

        return {
            "id": torch.LongTensor(ids),
            "net_input": {
                "src_tokens": torch.stack(collated_sources).float(),
                "encoder_padding_mask": torch.stack(padding_masks),
            }
        }

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            order = np.lexsort(
                [np.random.permutation(len(self)), np.array(self.sizes)]
            )
            return order[::-1]
        else:
            return np.arange(len(self))
