# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from pathlib import Path
from argparse import Namespace
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from fairseq.data import Dictionary
from fairseq.data import AddTargetDataset
from fairseq.data import LanguagePairDataset

from dataclasses import dataclass
from dataclasses import field
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from typing import Optional
from omegaconf import MISSING, II

from fairseq.data.sign_language import (
    SignFeatsType,
    SLTopicDetectionDataset,
)
from fairseq.data.text_compressor import TextCompressor
from fairseq.data.text_compressor import TextCompressionLevel
from fairseq.tasks import FairseqTask
from fairseq.tasks import register_task
from fairseq import metrics

logger = logging.getLogger(__name__)


@dataclass
class SLTopicDetectionConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    dict_path: str = field(
        default = MISSING,
        metadata={'help': 'Path to dictionary mapping category number to category name'},
    )
    modeling_task: str = field(
        default = 'classification',
        metadata={'help': 'Modeling task.'},
    )
    num_labels: str = field(
        default=10, metadata={'help': 'Number of labelswhen modeling_task is classification'}
    )
    max_source_positions: Optional[int] = field(
        default=5500, metadata={"help": "max number of frames in the source sequence"}
    )
    min_source_positions: Optional[int] = field(
        default=150, metadata={"help": "min number of frames in the source sequence"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    body_parts: str = field(
        default = "face,upperbody,lefthand,righthand",
        metadata={"help": "Select the keypoints that you want to use. Options: 'face','upperbody','lowerbody','lefthand', 'righthand'"},
    )
    feat_dims: str = field(
        default = "0,1,2",
        metadata={"help": "Select the keypoints dimensions that you want to use. Options: 0, 1, 2, 3"},
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "set True to shuffle the dataset between epochs"},
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )
    feats_type: ChoiceEnum([x.name for x in SignFeatsType]) = field(
        default="keypoints",
        metadata={
            "help": (
                "type of features for the sign input data:"
                "keypoints/mediapipe_keypoints/rotational/mediapipe_rotational/i3d/spot_align/spot_align_albert/mouthings/mouthings_albert/text/text_albert (default: keypoints)."
            )
        },
    )
    eval_accuracy: bool = field(
        default=True,
        metadata={'help': 'set to True to evaluate validation accuracy'},
    )
    tpu: bool = II("common.tpu")
    bpe_sentencepiece_model: str = II("bpe.sentencepiece_model")


@register_task("SL_topic_detection", dataclass=SLTopicDetectionConfig)
class SLTopicDetectionTask(FairseqTask):
    def __init__(self, cfg, label_dict=None, src_dict=None):  # TODO: check that src_dict is passed when text data is used
        super().__init__(cfg)
        self.label_dict = label_dict
        self.src_dict = src_dict
        if SignFeatsType[cfg.feats_type] in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings]:
            self.bpe_tokenizer = self.build_bpe(
                Namespace(
                    bpe='sentencepiece',
                    sentencepiece_model=cfg.bpe_sentencepiece_model
                )
            )
        self.softmax = nn.Softmax(dim=1)

    @classmethod
    def setup_task(cls, cfg):
        if 'SEED' in os.environ:
            seed = int(os.environ.get('SEED'))
            torch.manual_seed(seed)
            np.random.seed(seed)
        if SignFeatsType[cfg.feats_type] in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings]:
            # cfg.bpe_sentencepiece_model = os.environ.get('SP_MODEL', cfg.bpe_sentencepiece_model) ## TODO: this is a temporary fix for ALTI on transformerCLS
            dict_path = Path(cfg.bpe_sentencepiece_model).with_suffix('.txt')
            # print(f'dict_path = {dict_path}')
            if not dict_path.is_file():
                raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
            src_dict = Dictionary.load(dict_path.as_posix())
            logger.info(
                f"dictionary size ({dict_path.name}): " f"{len(src_dict):,}"
            )
            return cls(cfg, src_dict=src_dict)
        return cls(cfg)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):

        root_dir = Path(self.cfg.data)
        assert root_dir.is_dir(), f'{root_dir} does not exist'

        manifest_file = root_dir / f"{split}_filt.csv"
        if SignFeatsType(self.cfg.feats_type) in [
            SignFeatsType.keypoints, SignFeatsType.mediapipe_keypoints,
            SignFeatsType.rotational, SignFeatsType.mediapipe_rotational,
            SignFeatsType.i3d, SignFeatsType.spot_align_albert, SignFeatsType.text_albert, SignFeatsType.mouthings_albert
        ]:
            feats_path = root_dir / f"{split}_filt.h5"
        elif SignFeatsType(self.cfg.feats_type) == SignFeatsType.video:
            DATA_PATH = {
                'train': '/home/alvaro/Documents/ML and DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/video',
                'val': '/home/alvaro/Documents/ML and DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/video',
                'test': '/home/alvaro/Documents/ML and DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/video',
            }
            feats_path = DATA_PATH[split]
        # TODO: decide what path to load from. Probably: feats_path = root_dir / f"{split}_filt.h5"
        elif SignFeatsType(self.cfg.feats_type) in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings]:
            feats_path = None
        else:
            raise NotImplementedError(
                (
                    'Features other than i3d, keypoints, rotational, spot_align, spot_align_albert, mouthings, mouthings_albert text or text_albert'
                    'are not available for How2Sign yet'
                )
            )

        if self.cfg.num_batch_buckets > 0 or self.cfg.tpu:
            raise NotImplementedError("Pending to implement bucket_pad_length_dataset wrapper")

        print(f'manifest_file {manifest_file}', flush=True)

        self.datasets[split] = SLTopicDetectionDataset.from_manifest_file(
            manifest_file=manifest_file,
            feats_path=feats_path,
            feats_type=self.cfg.feats_type,
            bodyparts=self.cfg.body_parts.split(','),
            feat_dims=[int(d) for d in self.cfg.feat_dims.split(',')],
            min_sample_size=self.cfg.min_source_positions,
            max_sample_size=self.cfg.max_source_positions,
            shuffle=self.cfg.shuffle_dataset,
            normalize=self.cfg.normalize,
        )

        data = pd.read_csv(manifest_file, sep="\t")

        text_compressor = TextCompressor(level=self.cfg.text_compression_level)
        labels = [
            text_compressor.compress(str(row['CATEGORY_ID']))
            for _, row in data.iterrows()
            if row['VIDEO_ID'] not in self.datasets[split].skipped_ids
        ]

        assert len(labels) == len(self.datasets[split]), (
            f"The length of the labels list ({len(labels)}) and the dataset length"
            f" after skipping some ids ({len(self.datasets[split].skipped_ids)})"
            f" do not match. Original dataset length is ({len(self.datasets[split])})"
        )

        def process_label_fn(label):
            return torch.tensor([int(label)]) - 1

        def label_len_fn(label):
            return len(torch.tensor([int(label)]))

        if SignFeatsType(self.cfg.feats_type) in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings]:
            # TODO: decide if input text data should be compressed also
            def process_sentence_fn(sentence):
                tokens = self.source_dictionary.encode_line(
                            self.bpe_tokenizer.encode(sentence),
                            append_eos=False,
                            add_if_not_exist=False,
                        )
                return tokens

            def sentence_len_fn(tokens):
                return tokens.numel()

            sentences = [
                process_sentence_fn(row['TEXT'])
                for i, row in data.iterrows()
                if row['VIDEO_ID'] not in self.datasets[split].skipped_ids
            ]
            lengths = [sentence_len_fn(tokens) for tokens in sentences]

            assert len(sentences) == len(self.datasets[split]), (
                f"The length of the sentences list ({len(sentences)}) and the dataset's length"
                f" after skipping some ids ({len(self.datasets[split].skipped_ids)})"
                f" do not match. Original dataset length is ({len(self.datasets[split])})"
            )

            labels = [
                torch.tensor([int(row['CATEGORY_ID'])]) - 1
                for _, row in data.iterrows()
                if row['VIDEO_ID'] not in self.datasets[split].skipped_ids
            ]

            self.datasets[split] = LanguagePairDataset(
                src=sentences,
                src_sizes=lengths,
                src_dict=self.source_dictionary,
                tgt=labels,
                tgt_sizes=torch.ones(len(labels)),  # targets have length 1
                left_pad_source=False,
                # Since our target is a single class label, there's no need for
                # teacher forcing. If we set this to ``True`` then our Model's
                # ``forward()`` method would receive an additional argument called
                # *prev_output_tokens* that would contain a shifted version of the
                # target sequence.
                input_feeding=False,
                append_eos_to_target=False,
                eos=self.source_dictionary.eos(),
            )
        else:
            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=0,
                eos=None,
                batch_targets=True,
                process_label=process_label_fn,
                label_len_fn=label_len_fn,
                add_to_input=False,
            )

    @property
    def target_dictionary(self):
        return self.label_dict

    @property
    def source_dictionary(self):
        return self.src_dict

    def max_positions(self):
        return self.cfg.max_source_positions

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = []
        for l in lines:
            h5_file, _id = l.split(':')
            feats_path = h5py.File(h5_file, "r")
            n_frames.append(np.array(feats_path[_id]).shape[0])
        return lines, n_frames

    # TODO: Implement this method
    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        raise NotImplementedError
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

    #Add this for validation
    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg)
        if from_checkpoint:
            pass  # TODO: Implement this
        return model

    #Add this for validation
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_accuracy:
            model.eval()
            with torch.no_grad():
                out = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'])
                preds = torch.argmax(self.softmax(out), dim=1)

            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            targets = sample['target']
            logging_output['_acc_counts_'] = sum(
                torch.eq(
                    preds.flatten(),
                    targets.flatten()
                    )
                ).item()
            logging_output['_acc_totals_'] = targets.flatten().shape[0]
        return loss, sample_size, logging_output

    def inference_step(
        self, sample, model, output_attentions=None, targets_container=None, preds_container=None,
    ):
        model.eval()
        with torch.no_grad():
            if output_attentions:
                out = model(
                    sample['net_input']['src_tokens'],
                    sample['net_input']['src_lengths'],
                    output_attentions=output_attentions
                )
            else:
                out = model(
                    sample['net_input']['src_tokens'],
                    sample['net_input']['src_lengths']
                )
            preds = torch.argmax(self.softmax(out), dim=1)

        # we split counts into separate entries so that they can be
        # summed efficiently across workers using fast-stat-sync
        targets = sample['target']
        if targets_container is not None:
            targets_container.append(targets)
        if preds_container is not None:
            preds_container.append(preds)

        counts = sum(
            torch.eq(
                preds.flatten(),
                targets.flatten()
                )
            ).item()
        total = targets.flatten().shape[0]
        return counts, total

    #Add this for validation
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_accuracy:

            def sum_logs(key):
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            counts.append(sum_logs('_acc_counts_'))
            totals.append(sum_logs('_acc_totals_'))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_acc_counts_', np.array(counts))
                metrics.log_scalar('_acc_totals_', np.array(totals))

                def compute_accuracy(meters):
                    acc = meters['_acc_counts_'].sum[0] / meters['_acc_totals_'].sum[0]
                    return round(acc, 2)

                metrics.log_derived('acc', compute_accuracy)
