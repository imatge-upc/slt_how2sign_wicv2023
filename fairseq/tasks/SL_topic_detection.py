# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from pathlib import Path
from typing import Optional
from argparse import Namespace
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from omegaconf import MISSING, II

import torch
import torch.nn as nn

from fairseq import metrics, utils
from fairseq.data import Dictionary, AddTargetDataset
from fairseq.data.sign_language import SignFeatsType, SignFeatsDataset, NormType
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum

from fairseq.tasks import FairseqTask, register_task

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
    max_target_positions: Optional[int] = field(
        default=1, metadata={"help": "max number of tokens in the target sequence, for TD it must be one"}
    )
    normalization: ChoiceEnum([x.name for x in NormType]) = field(
        default=NormType.body.name,
        metadata={"help": "select the type of normalization to apply"},
    )
    data_augmentation: bool = field(
        default=False,
        metadata={"help": "set True to apply data_augmentation to every sample"},
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "set True to shuffle the dataset between epochs"},
    )
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )
    feats_type: ChoiceEnum([x.name for x in SignFeatsType]) = field(
        default="i3d",
        metadata={
            "help": (
                "type of features for the sign input data: mediapipe/i3d/CNN2d/openpose (default: i3d). "
            )
        },
    )
    eval_accuracy: bool = field(
        default=True,
        metadata={'help': 'set to True to evaluate validation accuracy'},
    )
    #Inherit from other configs
    train_subset: str = II("dataset.train_subset")
    valid_subset: str = II("dataset.valid_subset")
    bpe_sentencepiece_model: str = II("bpe.sentencepiece_model")


@register_task("SL_topic_detection", dataclass=SLTopicDetectionConfig)
class SLTopicDetectionTask(FairseqTask):
    def __init__(self, cfg, label_dict=None, src_dict=None):
        super().__init__(cfg)
        self.label_dict = label_dict
        self.src_dict = src_dict #Only for input text data
        '''
        if SignFeatsType_TD[cfg.feats_type] in [SignFeatsType_TD.text, SignFeatsType_TD.spot_align, SignFeatsType_TD.mouthings]:
            self.bpe_tokenizer = self.build_bpe(
                Namespace(
                    bpe='sentencepiece',
                    sentencepiece_model=cfg.bpe_sentencepiece_model
                )
            )'''
        self.softmax = nn.Softmax(dim=1)

    @classmethod
    def setup_task(cls, cfg):
        if 'SEED' in os.environ:
            seed = int(os.environ.get('SEED'))
            torch.manual_seed(seed)
            np.random.seed(seed)
        '''
        if SignFeatsType_TD[cfg.feats_type] in [SignFeatsType_TD.text, SignFeatsType_TD.spot_align, SignFeatsType_TD.mouthings]:
            # cfg.bpe_sentencepiece_model = os.environ.get('SP_MODEL', cfg.bpe_sentencepiece_model) ## TODO: this is a temporary fix for ALTI on transformerCLS
            dict_path = Path(cfg.bpe_sentencepiece_model).with_suffix('.txt')
            # print(f'dict_path = {dict_path}')
            if not dict_path.is_file():
                raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
            src_dict = Dictionary.load(dict_path.as_posix())
            logger.info(
                f"dictionary size ({dict_path.name}): " f"{len(src_dict):,}"
            )
            return cls(cfg, src_dict=src_dict)'''
        if getattr(cfg, "train_subset", None) is not None:
            if not all("train" in s for s in cfg.train_subset.split(",")):
                raise ValueError('Train splits should be containe the word "train".')
        return cls(cfg)
        
    def load_dataset(self, split, epoch=1, combine=False, task_cfg: FairseqDataclass = None, **kwargs):
        is_train_split = "train" in split
        
        root_dir = Path(self.cfg.data)
        assert root_dir.is_dir(), f'{root_dir} does not exist'
        manifest_file = root_dir / f"{split}.tsv"
        
        '''
        manifest_file = root_dir / f"{split}_filt.csv"
        if SignFeatsType_TD(self.cfg.feats_type) in [
            SignFeatsType_TD.keypoints, SignFeatsType_TD.mediapipe_keypoints,
            SignFeatsType_TD.rotational, SignFeatsType_TD.mediapipe_rotational,
            SignFeatsType_TD.i3d, SignFeatsType_TD.spot_align_albert, SignFeatsType_TD.text_albert, SignFeatsType_TD.mouthings_albert
        ]:
            feats_path = root_dir / f"{split}_filt.h5"
        elif SignFeatsType_TD(self.cfg.feats_type) == SignFeatsType_TD.video:
            DATA_PATH = {
                'train': '/home/alvaro/Documents/ML and DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/video',
                'val': '/home/alvaro/Documents/ML and DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/video',
                'test': '/home/alvaro/Documents/ML and DL/How2Sign/TFG/Sign-Language-Topic-Detection/data/How2Sign/video',
            }
            feats_path = DATA_PATH[split]
        # TODO: decide what path to load from. Probably: feats_path = root_dir / f"{split}_filt.h5"
        elif SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.text, SignFeatsType_TD.spot_align, SignFeatsType_TD.mouthings]:
            feats_path = None
        else:
            raise NotImplementedError(
                (
                    'Features other than i3d, keypoints, rotational, spot_align, spot_align_albert, mouthings, mouthings_albert text or text_albert'
                    'are not available for How2Sign yet'
                )
            )
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
        '''
        
        
        self.datasets[split] = SignFeatsDataset.from_manifest_file(
            manifest_file=manifest_file,
            normalization=self.cfg.normalization,
            data_augmentation=(self.cfg.data_augmentation and is_train_split),
            min_sample_size=self.cfg.min_source_positions,
            max_sample_size=self.cfg.max_source_positions,
            shuffle=self.cfg.shuffle_dataset,
        )
        
        if is_train_split:
            self.datasets[split].filter_by_length(
                self.cfg.min_source_positions,
                self.cfg.max_source_positions,
            )
        data = pd.read_csv(manifest_file, sep="\t")

        text_compressor = TextCompressor(level=self.cfg.text_compression_level)
        labels = [
            text_compressor.compress(str(row['topic']))
            for _, row in data.iterrows()
            if row['id'] not in self.datasets[split].skipped_ids
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
        '''
        if SignFeatsType_TD(self.cfg.feats_type) in [SignFeatsType_TD.text, SignFeatsType_TD.spot_align, SignFeatsType_TD.mouthings]:
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
        else:'''
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
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    '''def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = []
        for l in lines:
            h5_file, _id = l.split(':')
            feats_path = h5py.File(h5_file, "r")
            n_frames.append(np.array(feats_path[_id]).shape[0])
        return lines, n_frames '''

    # TODO: Implement this method
    '''def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        raise NotImplementedError
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )'''

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
                out = model(sample['net_input']['src_tokens'], sample['net_input']['encoder_padding_mask'])
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
