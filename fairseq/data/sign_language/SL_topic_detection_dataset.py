import os
import sys
import logging
from enum import Enum
from pathlib import Path
from typing import List, Union, Optional

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from fairseq.data import FairseqDataset

from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

logger = logging.getLogger(__name__)


class SignFeatsType_TD(Enum):
    text = "text"
    text_albert = "text_albert"
    spot_align = "spot_align"
    mouthings = "mouthings"
    spot_align_albert = "spot_align_albert"
    mouthings_albert = "mouthings_albert"
    keypoints = "keypoints"
    mediapipe_keypoints = "mediapipe_keypoints"
    rotational = "rotational"
    mediapipe_rotational = "mediapipe_rotational"
    i3d = "i3d"
    CNN2d = "CNN2d"
    video = 'video'

class SLTopicDetectionDataset(FairseqDataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        ids: List[str],
        feats_path: Union[Path, str],
        feats_type: str,
        sizes: List[int] = None,
        bodyparts: Optional[List[str]] = None,
        feat_dims: List[int] = [0, 1, 2, 3],
        min_sample_size: int = 0,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        normalize: bool = False,
        text_compression_level: TextCompressionLevel = TextCompressionLevel.none,
    ):
        super().__init__()
        self.text_compressor = TextCompressor(level=text_compression_level)

        self.manifest = manifest

        # if feats_type == SignFeatsType.video, feats_path is the directory where .mp4 files of the corresponding split are stored
        self.feats_path = feats_path
        self.ids = [_id for _id in ids]

        if feats_type not in ['video']:
            if feats_type in ['text', 'spot_align', 'mouthings']:
                self.feats_file = self.manifest.set_index('VIDEO_ID').to_dict()['TEXT']
            else:
                self.feats_file = h5py.File(self.feats_path, 'r')
                if sizes is None:
                    sizes = []
                    for _id in self.ids:
                        _id = _id
                        sizes.append(np.array(self.feats_file[_id]).shape[0])
        self.sizes = sizes

        self.feats_type = feats_type
        self.bodyparts = bodyparts
        self.feat_dims = feat_dims

        self.shuffle = shuffle
        self.normalize = normalize

        self.min_sample_size = min_sample_size
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.skipped_ids = []
        for _id, size in zip(self.ids[:], self.sizes[:]):
            if size < self.min_sample_size or size > self.max_sample_size:
                self.sizes.pop(self.ids.index(_id))
                self.ids.remove(_id)
                self.skipped_ids.append(_id)
        logger.info(f"Skipped {len(self.skipped_ids)} input sequences, that were either too short or too long.")

        try:
            import pyarrow as pa
            self.ids = pa.array(self.ids)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

    @staticmethod
    def list_avail_ids(self):
        return self.ids

    @classmethod
    def from_manifest_file(cls, manifest_file: Union[str, Path], **kwargs):
        ids = []
        sizes = []
        manifest = pd.read_csv(manifest_file, sep="\t")
        for _, row in manifest.iterrows():
            ids.append(row['VIDEO_ID'])
            size = int(row['END_FRAME']) - int(row['START_FRAME']) + 1
            sizes.append(size)
        logger.info(f"loaded {len(ids)} samples")
        return cls(manifest, ids, sizes=sizes, **kwargs)

    def __getitem__(self, index):
        _id = self.ids[index]
        _id = _id if isinstance(self.ids, list) else _id.as_py()
        fn = _id
        if self.feats_type in ['video']:  # load corresponding mp4
            import torchvision
            # there is no repeated value in column VIDEO_ID of self.manifest
            video_name = self.manifest[self.manifest.VIDEO_ID.str.match(fn)]['VIDEO_NAME'].values[0]
            feats = torchvision.io.read_video(filename=os.path.join(self.feats_path, video_name + '.mp4'), end_pts=5115)[0]
            feats = feats.permute(0, 3, 1, 2)
        elif self.feats_type in ['text', 'spot_align', 'mouthings']:
            feats = torch.Tensor(np.array(self.feats_file[fn]))
        else:
            feats = torch.Tensor(np.array(self.feats_file[fn])).float()
        feats = self.postprocess(feats)

        return {"id": index, "h2s_id": fn, "source": feats}

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats):
        from fairseq.data.sign_language.utils import (
            select_keypoints_by_bodypart,
            select_keypoints_by_dimension,
        )
        if SignFeatsType[self.feats_type] in [SignFeatsType.keypoints, SignFeatsType.mediapipe_keypoints]:
            feats, n_feats = select_keypoints_by_bodypart(feats, feats_type=self.feats_type, bodyparts=self.bodyparts)
            feats = select_keypoints_by_dimension(feats, self.feat_dims, feats_type=self.feats_type)
            feats_split = feats.reshape(-1, n_feats, 3).permute(2, 0, 1)
            with torch.no_grad():
                feats_norm_split = F.layer_norm(feats_split, feats_split.shape[1:])
            feats = feats_norm_split.permute(1, 2, 0).reshape(-1, n_feats * 3).contiguous()
        elif SignFeatsType[self.feats_type] in [SignFeatsType.rotational, SignFeatsType.mediapipe_rotational]:
            feats_split = feats.reshape(-1, 48, 6).permute(2, 0, 1)
            with torch.no_grad():
                feats_norm_split = F.layer_norm(feats_split, feats_split.shape[1:])
            feats = feats_norm_split.permute(1, 2, 0).reshape(-1, 48 * 6).contiguous()
        elif (SignFeatsType[self.feats_type] is SignFeatsType.i3d or
              SignFeatsType[self.feats_type] is SignFeatsType.CNN2d or
              SignFeatsType[self.feats_type] is SignFeatsType.video or
              SignFeatsType[self.feats_type] is SignFeatsType.spot_align_albert or
              SignFeatsType[self.feats_type] is SignFeatsType.mouthings_albert or
              SignFeatsType[self.feats_type] is SignFeatsType.text_albert):
            with torch.no_grad():
                feats = F.layer_norm(feats.float(), feats.shape)
        elif SignFeatsType[self.feats_type] in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings]:
            pass
        else:
            raise NotImplementedError(f'Using {self.feats_type} which is not SignFeatsType.i3d'
                                      ' nor SignFeatsType.spot_align_albert'
                                      ' nor SignFeatsType.mouthings_albert'
                                      ' nor SignFeatsType.keypoints nor SignFeatsType.mediapipe_keypoints'
                                      ' nor SignFeatsType.rotational nor SignFeatsType.mediapipe_rotational'
                                      ' nor SignFeatsType.2dCNN nor SignFeatsType.video'
                                      ' nor SignFeatsType.text nor SignFeatsType.spot_align'
                                      ' nor SignFeatsType.text nor SignFeatsType.mouthings'
                                      )
        return feats

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.feats_type not in ['video']:
            collated_sources = sources[0].new_zeros(len(sources), max(sizes), sources[0].shape[-1])
        else:
            collated_sources = sources[0].new_zeros(len(sources), max(sizes), *sources[0].shape[-3:])
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - max(sizes)
            if self.feats_type not in ['video']:
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff, source.shape[-1]), 0.0)]
                )
            else:
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff, *source.shape[-3:]), 0.0)]
                )
        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'net_input': {
                'src_tokens': collated_sources, 
                'src_lengths': torch.Tensor(sizes)  # FIXME: If you use buckets
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
