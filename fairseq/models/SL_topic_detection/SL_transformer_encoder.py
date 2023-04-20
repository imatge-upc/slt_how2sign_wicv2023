#!/usr/bin/env python3

from json import encoder
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.data.sign_language.utils import get_num_feats
from fairseq.models import (
    FairseqEncoder,
    BaseFairseqModel,
    register_model,
)

from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from fairseq.models.transformer import Embedding

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from omegaconf import II

from fairseq.data.sign_language import SignFeatsType
logger = logging.getLogger(__name__)


class Conv1dSubsampler(nn.Module):
    '''
    Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
        strides (List[int]): the stride for each convolutional layer
    '''
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
        strides: List[int] = (2, 2),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.kernel_sizes=kernel_sizes
        self.n_layers = len(kernel_sizes)
        assert len(kernel_sizes) == len(strides)
        self.strides = strides
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                s,
                padding=k // 2,
            )
            for i, (k, s) in enumerate(zip(kernel_sizes, strides))
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for s in self.strides:
            out = ((out.float() - 1) / s + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        x = src_tokens.transpose(1, 2).contiguous()
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x, self.get_out_seq_lens_tensor(src_lengths)

@dataclass
class SLTopicDetectionTransformerConfig_alvaro(FairseqDataclass):
    '''
    Add model-specific arguments to the parser.
    '''
    # input
    subsample_input: bool = field(
        default=False, metadata={'help': 'if True subsample inputs along index (temporal) dimension'}
    )
    conv_kernel_sizes: str = field(
        default="5,5", metadata={"help": "kernel sizes of Conv1d subsampling layers"}
    )
    conv_strides: str = field(
        default="3,3", metadata={"help": "stride of Conv1d subsampling layers"}
    )
    conv_channels: int = field(
        default=1024, metadata={"help": "# of channels in Conv1d subsampling layers"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability after activation in FFN."}
    )
    encoder_embed_dim: int = field(
        default=512, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers"}
    )
    encoder_attention_heads: Optional[int] = field(
        default=None, metadata={"help": "num encoder attention heads"}
    )
    encoder_normalize_before: bool = field(
        default=True, metadata={"help": "apply layernorm before each encoder block"}
    )
    load_pretrained_encoder_from: str = field(
        default="relu", metadata={"help": "model to take encoder weights from (for initialization)"}
    )
    encoder_freezing_updates: int = field(
        default=0, metadata={"help": "freeze encoder for first N updates"}
    )
    feats_type: ChoiceEnum([x.name for x in SignFeatsType]) = II("task.feats_type")
    body_parts: str = II("task.body_parts")
    feat_dims: str = II("task.feat_dims")
    max_source_positions: int = II("task.max_source_positions")


@register_model("SL_topic_detection_transformer_alvaro", dataclass=SLTopicDetectionTransformerConfig_alvaro)
class Sign2TextTransformerModel(BaseFairseqModel):
    '''
    Adapted Transformer model for SL Topic Detection tasks. The Transformer
    encoder remains the same as in "Attention is All You Need".
    A trainable input subsampler is prepended to the Transformer encoder
    to downsample the input sequences to a manageable length..
    '''
    @classmethod
    def hub_models(cls):
        base_url = "" # TODO: Set base URL to upload checkpoints
        model_ids = [
            'SL_topic_detection_transformer_s-how2sign',
            'SL_topic_detection_transformer_m-how2sign',
            'SL_topic_detection_transformer_l-how2sign',
        ]
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    # TODO: Check this
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file='model.pt',
        data_name_or_path='.',
        config_yaml='config.yaml',
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config_yaml,
            **kwargs,
        )
        return S2THubInterface(x['cfg'], x['task'], x['models'][0]) #Aquí s'utilitza, hauriem de mirar què necessitem i que es cadascun dels arguments d'aquestes.

    def __init__(self, cfg, encoder, att_encoder, classif_head):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.att_encoder = att_encoder
        self.classif_head = classif_head

    @classmethod
    def build_encoder(cls, cfg, encoder_embed_tokens=None):
        # TODO: see how to pass this encoder_embed_tokens to the encoder
        encoder = SLTopicDetectionTransformerEncoder(cfg, encoder_embed_tokens)
        pretraining_path = getattr(cfg, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_classif_head(cls, cfg):
        classif_head = SLTopicDetectionClassifHead(cfg)
        pretraining_path = getattr(cfg, "load_pretrained_classif_head_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                classif_head = checkpoint_utils.load_pretrained_component_from_model(
                    component=classif_head, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained classification head from: {pretraining_path}")
        return classif_head

    @classmethod
    def build_att_encoder(cls, cfg):
        att_encoder = EncoderAttentionLayer(cfg)
        pretraining_path = getattr(cfg, "load_pretrained_att_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                classif_head = checkpoint_utils.load_pretrained_component_from_model(
                    component=classif_head, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained attention encoder layer from: {pretraining_path}")
        return att_encoder

    @classmethod
    def build_model(cls, cfg, task):
        '''
        Build a new model instance.
        '''
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        encoder_embed_tokens = None
        if SignFeatsType[cfg.feats_type] in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings]:
            encoder_embed_tokens = build_embedding(
                task.source_dictionary, cfg.encoder_embed_dim
            )
        encoder = cls.build_encoder(cfg, encoder_embed_tokens)

        print(cfg.keys(), flush=True)

        att_encoder = cls.build_att_encoder(cfg)
        classif_head = cls.build_classif_head(cfg)

        return cls(cfg, encoder, att_encoder, classif_head)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths):
        '''
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        '''
        x = self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths
        )

        x = self.att_encoder(x['encoder_out'][0])[0]  # ignore attention weights and keep embeddings
        return self.classif_head(x)


class SLTopicDetectionTransformerEncoder(FairseqEncoder):
    '''
    Transformer encoder that consists of (optional) input subsampler and Transformer encoder.
    '''
    def __init__(self, cfg, encoder_embed_tokens=None):
        super().__init__(None)
        self.encoder_freezing_updates = cfg.encoder_freezing_updates
        self.num_updates = 0

        self.encoder_embed_tokens = encoder_embed_tokens
        self.dropout_module = FairseqDropout(
            p=cfg.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(cfg.encoder_embed_dim)
        self.padding_idx = 1

        self.feats_type = SignFeatsType[cfg.feats_type]
        self.subsample = None

        if cfg.subsample_input:
            self.subsample = Conv1dSubsampler(
                get_num_feats(
                    SignFeatsType[cfg.feats_type],
                    cfg.body_parts.split(','),
                    cfg.feat_dims.split(',')
                ) if self.feats_type not in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings] else cfg.encoder_embed_dim,
                cfg.conv_channels,
                cfg.encoder_embed_dim,
                [int(k) for k in cfg.conv_kernel_sizes.split(",")],
                [int(k) for k in cfg.conv_strides.split(",")],
            )
        else:
            cfg.encoder_embed_dim = get_num_feats(
                SignFeatsType[cfg.feats_type],
                cfg.body_parts.split(','),
                cfg.feat_dims.split(','),
            )
            if self.feats_type not in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings]:
                TypeError(f'When cfg.subsample_input=True, feats_type is expected to be `text` or `spot_align`, `mouthings` but got {self.feats_type} instead!')

        self.embed_positions = PositionalEmbedding(
            cfg.max_source_positions,
            cfg.encoder_embed_dim,
            self.padding_idx,
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        )
        if cfg.encoder_normalize_before:
            self.layer_norm = LayerNorm(cfg.encoder_embed_dim)
        else:
            self.layer_norm = None

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        x = src_tokens  # B x T
        if self.feats_type in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings]:
            x = self.encoder_embed_tokens(x)
        if self.subsample:
            x, input_lengths = self.subsample(x, src_lengths)
        else:
            x, input_lengths = x.transpose(0, 1), src_lengths.long()
        #  x: T x B x C

        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions

        x = self.dropout_module(x)

        encoder_states = []
        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
            "input_lengths": [input_lengths],
        }

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
                )
        else:
            x = self._forward(
                src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
            )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

class SLTopicDetectionClassifHead(BaseFairseqModel):
    '''
    Classification head
    '''
    def __init__(self, cfg):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(
                cfg.encoder_embed_dim,
                cfg.encoder_embed_dim // 2,
            ),
            nn.ReLU(),
            nn.Linear(cfg.encoder_embed_dim // 2, 10),
        )

    def forward(self, x):
        # x: batch x hidden
        return self.classifier(x)


class EncoderAttentionLayer(BaseFairseqModel):
    '''
    Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
    https://aclanthology.org/P16-2034/
    Projects the input vectors and then performs a weighted sum of the input vectors.
    '''
    def __init__(self, cfg):
        super().__init__()
        self.w = torch.nn.parameter.Parameter(
            data=torch.randn(
                    cfg.encoder_embed_dim
            ),
            requires_grad=True,
        )
        stdv = 1. / math.sqrt(self.w.size(0))
        self.w.data.uniform_(-stdv, stdv)

        self.tanh = nn.Tanh()
        self.softm = nn.Softmax(dim=1)

    def forward(self, H):
        # H: seq_len x batch x hidden
        H = H.permute(1, 2, 0)
        # H: batch x hidden x seq_len

        M = self.tanh(H)
        # M: batch x hidden x seq_len

        alpha = self.softm(torch.matmul(self.w, M))
        # alpha: batch x seq_len

        r = torch.matmul(H, alpha.unsqueeze(-1)).squeeze(-1)
        # r: batch x hidden
        return r, alpha
