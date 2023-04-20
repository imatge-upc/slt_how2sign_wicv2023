import sys
import logging
from enum import Enum
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from pose_format import Pose

from fairseq.data import FairseqDataset, BaseWrapperDataset, RandomCropDataset

from fairseq.data.data_utils import (
    compute_mask_indices,
    numpy_seed
)

from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

logger = logging.getLogger(__name__)

class SignFeatsType(str, Enum):
    mediapipe = "mediapipe"
    openpose = "openpose"
    i3d = "i3d"
    CNN2d = "CNN2d"

class NormType(str, Enum):
    body = "body"
    kp_wise = "kp_wise"
    global_xyz = "global_xyz"
    normalize = "normalize" #to add the same normalizaiton as original TD

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
        
        if self.feats_type == SignFeatsType.mediapipe:
            with open(feats_file, "rb") as f:
                pose = Pose.read(f.read())
            frames_list = list(range(offset, offset+length))

            # Fix to bypass some examples that are wrong
            frames_list = [fr for fr in frames_list if fr < pose.body.data.shape[0]]

            pose.body = pose.body.select_frames(frames_list)
            pose = self.postprocess(pose)
        elif self.feats_type == SignFeatsType.i3d or self.feats_type == SignFeatsType.openpose:
            with open(feats_file, "rb") as f:
                pose = np.load(f)
            pose = self.postprocess(pose)

        return {"id": index, "vid_id": _id, "source": pose}

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, pose):
        
        if SignFeatsType[self.feats_type] in [SignFeatsType.mediapipe, SignFeatsType.openpose]:
            import mediapipe as mp
            mp_holistic = mp.solutions.holistic
            FACEMESH_CONTOURS_POINTS = [
                str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))
            ]
            POSE_RM = ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
                    'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
            POSE_POINTS = [kp.name for kp in mp_holistic.PoseLandmark if kp.name not in POSE_RM]
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
                pass
            if self.data_augmentation:
                pose = pose.augment2d()
            pose = pose.torch()
                
        elif (SignFeatsType[self.feats_type] in [SignFeatsType.i3d, SignFeatsType.CNN2d]):
            pose = torch.from_numpy(pose)
        else:
            raise NotImplementedError(f'Using {self.feats_type} which is not SignFeatsType.i3d'
                                      ' nor SignFeatsType.mediapipe nor SignFeatsType.openpose'
                                      ' nor SignFeatsType.2dCNN '
                                      )
        return pose

    def collater(self, samples):
        if self.feats_type == SignFeatsType.mediapipe:
            max_length = max([s['source'].body.data.shape[0] for s in samples])
        elif (self.feats_type == SignFeatsType.i3d) or (self.feats_type == SignFeatsType.openpose):
            max_length = max([s['source'].shape[0] for s in samples])
        
        ids = []
        padding_masks = []
        collated_sources = []
        for sample in samples:
            pose = sample['source']

            if self.feats_type == SignFeatsType.mediapipe:
                if pose.body.data.shape[1] > 1:
                    logger.warning(f"More than one person in frame, keeping just the first one")
                pose.body.data = pose.body.data[:, 0]

                padding_mask = (~pose.body.data.mask).sum((1,2)) > 0
            
            elif self.feats_type == SignFeatsType.i3d:
                padding_mask = torch.zeros(pose.shape[0], dtype=torch.bool)
            
            if padding_mask.all():
                continue

            diff_length = max_length - len(padding_mask)
            ids.append(sample['id'])
            padding_masks.append(
                F.pad(padding_mask, (0, diff_length), value=True)
            )
            
            if self.feats_type == SignFeatsType.mediapipe: 
                collated_sources.append(
                    F.pad(pose.body.data.data, (0, 0, 0, 0, 0, diff_length), value=0)
                )
            elif self.feats_type == SignFeatsType.i3d:
                collated_sources.append(
                    F.pad(pose, (0, 0, 0, diff_length), value=0)
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

# TODO: In task, if compute_mask_indices=True, create dataset of this type
# TODO: In task, if using this, it may be useful to wrap it also with RandomCropSignFeatsDataset (remember paddings)
class MaskSignFeatsDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: SignFeatsDataset,
        **mask_compute_kwargs,
        ):
        super().__init__(dataset)
        self.mask_compute_kwargs = mask_compute_kwargs
        self._features_size_map = {}
        self._C = mask_compute_kwargs["encoder_embed_dim"]
        self._conv_feature_layers = eval(mask_compute_kwargs["conv_feature_layers"])

    def _compute_mask_indices(self, dims, padding_mask):
        # Create masks for Sign2vec pretraining
        raise NotImplementedError("This feature is still not available")
        B, T, C = dims
        mask_indices, mask_channel_indices = None, None
        if self.mask_compute_kwargs["mask_prob"] > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_compute_kwargs["mask_prob"],
                self.mask_compute_kwargs["mask_length"],
                self.mask_compute_kwargs["mask_selection"],
                self.mask_compute_kwargs["mask_other"],
                min_masks=2,
                no_overlap=self.mask_compute_kwargs["no_mask_overlap"],
                min_space=self.mask_compute_kwargs["mask_min_space"],
            )
            mask_indices = torch.from_numpy(mask_indices)
        if self.mask_compute_kwargs["mask_channel_prob"] > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_compute_kwargs["mask_channel_prob"],
                self.mask_compute_kwargs["mask_channel_length"],
                self.mask_compute_kwargs["mask_channel_selection"],
                self.mask_compute_kwargs["mask_channel_other"],
                no_overlap=self.mask_compute_kwargs["no_mask_channel_overlap"],
                min_space=self.mask_compute_kwargs["mask_channel_min_space"],
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            )

        return mask_indices, mask_channel_indices

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        raise NotImplementedError("This feature is still not available")
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in self._conv_feature_layers:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def collater(self, samples):
        out = self.dataset.collater(samples)
        raise NotImplementedError("This feature is still not available")

        B = out["net_input"]["source"].size(0)
        T = self._get_mask_indices_dims(out["net_input"]["source"].size(-2))
        padding_mask_reshaped = out["net_input"]["padding_mask"].clone()
        extra = padding_mask_reshaped.size(1) % T
        if extra > 0:
            padding_mask_reshaped = padding_mask_reshaped[:, :-extra]
        padding_mask_reshaped = padding_mask_reshaped.view(
            padding_mask_reshaped.size(0), T, -1
        )
        padding_mask_reshaped = padding_mask_reshaped.all(-1)
        out["net_input"]["padding_count"] = padding_mask_reshaped.sum(-1).max().item()
        mask_indices, mask_channel_indices = self._compute_mask_indices(
            (B, T, self._C),
            padding_mask_reshaped,
        )
        out["net_input"]["mask_indices"] = mask_indices
        out["net_input"]["mask_channel_indices"] = mask_channel_indices
        out["sample_size"] = mask_indices.sum().item()
        
        return out


class RandomCropSignFeatsDataset(RandomCropDataset):
    def __init__(
        self,
        dataset: SignFeatsDataset,
        truncation_length: int,
        **kwargs,
    ):
        super().__init__(dataset, truncation_length, **kwargs)

    def __getitem__(self, index):
        with numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            item_len = item["source"].size(0)
            excess = item_len - self.truncation_length
            if excess > 0:
                start_idx = np.random.randint(0, excess)
                item["source"] = item["source"][start_idx : start_idx + self.truncation_length]
            return item