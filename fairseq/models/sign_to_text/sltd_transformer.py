#!/usr/bin/env python3

#from json import encoder
import logging
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from omegaconf import II

import torch
import torch.nn as nn
from torch import Tensor

from pose_format import Pose

from fairseq import checkpoint_utils, utils

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.data.sign_language import SignFeatsType
#from fairseq.data.sign_language.utils import get_num_feats

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum

from fairseq.models import (
    FairseqEncoder,
    BaseFairseqModel,
    register_model,
)

#from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.transformer import Embedding

from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)

logger = logging.getLogger(__name__)

@dataclass
class SLTopicDetectionTransformerConfig(FairseqDataclass):
    '''
    Add model-specific arguments to the parser.
    '''
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
    #body_parts: str = II("task.body_parts")
    #feat_dims: str = II("task.feat_dims")
    max_source_positions: int = II("task.max_source_positions")


@register_model("SL_topic_detection_transformer", dataclass=SLTopicDetectionTransformerConfig)
class Sign2TextTransformerModel(BaseFairseqModel):
    '''
    Adapted Transformer model for SL Topic Detection tasks. The Transformer
    encoder remains the same as in "Attention is All You Need".
    '''
    def __init__(self, cfg, encoder, att_encoder, classif_head):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.att_encoder = att_encoder
        self.classif_head = classif_head

    @classmethod
    def build_encoder(cls, cfg, feats_type, feat_dim):
        encoder = SLTopicDetectionTransformerEncoder(cfg, feats_type, feat_dim)
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
        '''This is going to be our decoder that predicts the topic.'''
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
        ''' Build a new model instance.'''
        if cfg.feats_type == SignFeatsType.i3d:
            feat_dim = 1024
        elif cfg.feats_type == SignFeatsType.mediapipe:
            feat_dim = 195
        elif cfg.feats_type == SignFeatsType.openpose:
            feat_dim = 150
        
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)
            
        encoder = cls.build_encoder(cfg, cfg.feats_type, feat_dim)

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

    def forward(self, src_tokens, encoder_padding_mask):
        '''
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        '''
        x = self.encoder(
            src_tokens=src_tokens,
            encoder_padding_mask=encoder_padding_mask
        )

        x = self.att_encoder(x['encoder_out'][0])[0]  # ignore attention weights and keep embeddings
        return self.classif_head(x)


class SLTopicDetectionTransformerEncoder(FairseqEncoder):
    '''
    Transformer encoder that consists of a Transformer encoder.
    '''
    def __init__(self, cfg, feats_type: SignFeatsType, feat_dim: int):
        super().__init__(None)
        #self.encoder_freezing_updates = cfg.encoder_freezing_updates
        self.num_updates = 0

        #self.encoder_embed_tokens = encoder_embed_tokens
        
        self.dropout_module = FairseqDropout(
            p=cfg.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(cfg.encoder_embed_dim)
        #if cfg.no_scale_embedding:
        #    self.embed_scale = 1.0
        self.padding_idx = 1

        self.feats_type = feats_type
        if feats_type == SignFeatsType.mediapipe or feats_type == SignFeatsType.openpose:
            self.feat_proj = nn.Linear(feat_dim * 3, cfg.encoder_embed_dim)
        if feats_type == SignFeatsType.i3d:
            self.feat_proj = nn.Linear(feat_dim, cfg.encoder_embed_dim)
        '''
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
        '''
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

    def forward(self, src_tokens, encoder_padding_mask, return_all_hiddens=False):
        if self.feats_type == SignFeatsType.mediapipe: #TODO: check what happens with openpose
            src_tokens = src_tokens.view(src_tokens.shape[0], src_tokens.shape[1], -1) #src_tokens B x seq_len x Fs          
        x = self.feat_proj(src_tokens).transpose(0, 1) #[seq_len, batch_size, embed_dim]
        # x: seq_len x B x H
        x = self.embed_scale * x

        #encoder_padding_mask = lengths_to_padding_mask(input_lengths)
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
            #"src_lengths": [src_lengths],
            #"input_lengths": [input_lengths],
        }
    '''
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
        return x'''

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
            #"src_lengths": [],  # B x 1
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
