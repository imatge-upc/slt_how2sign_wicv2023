#!/usr/bin/env python3

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
    FairseqEncoderDecoderModel,
    register_model,
)
from fairseq.models.speech_to_text.hub_interface import S2THubInterface #això haurem de veure en què s'utilitza i com ho fem nosaltres, si utilitzem el mateix.
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from typing import Optional, Any
from omegaconf import MISSING, II

from fairseq.data.sign_language import SignFeatsType
logger = logging.getLogger(__name__)


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
        strides (List[int]): the stride for each convolutional layer
    """

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
class Sign2TextTransformerConfig(FairseqDataclass):
    """Add model-specific arguments to the parser."""
    # input
    conv_kernel_sizes: str = field(
        default="5,5", metadata={"help": "kernel sizes of Conv1d subsampling layers"}
    )
    conv_strides: str = field(
        default="2,2", metadata={"help": "stride of Conv1d subsampling layers"}
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
        #default=8, metadata={"help": "num encoder attention heads"}
    )
    encoder_normalize_before: bool = field(
        default=True, metadata={"help": "apply layernorm before each encoder block"}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num decoder layers"}
    )
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=True, metadata={"help": "apply layernorm before each decoder block"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
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
    """
    decoder_learned_pos: bool = field(
        default=False, metadata={"help": "use learned positional embeddings"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if True, disables positional embeddings (outside self attention)"
        },
    )
    # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    decoder_layerdrop: float = field(
        default=0, metadata={"help": "Decoder LayerDrop probability"}
    )
    decoder_input_dim: int = field(
        default=II("model.decoder.embed_dim"),
        metadata={
            "help": "decoder input dimension (extra linear layer if different from decoder embed dim)"
        },
    )
    decoder_output_dim: int = field(
        default=II("model.decoder.embed_dim"),
        metadata={
            "help": "decoder output dimension (extra linear layer if different from decoder embed dim)"
        },
    )

    adaptive_input: bool = False
    adaptive_softmax_cutoff: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "list of adaptive softmax cutoff points. Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0.0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )

    quant_noise_pq: float = field(default=0)
    """

@register_model("sign2text_transformer", dataclass=Sign2TextTransformerConfig)
class Sign2TextTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model for sign-to-text tasks. The Transformer
    encoder/decoder remains the same. A trainable input subsampler is
    prepended to the Transformer encoder to project inputs into the encoder
    dimension as well as downsample input sequence for computational
    efficiency."""

    @classmethod
    def hub_models(cls):
        base_url = "" # TODO: Set base URL to upload checkpoints
        model_ids = [
            "sign2t_transformer_s-asl-en-how2sign",
            "sign2t_transformer_m-asl-en-how2sign",
            "sign2t_transformer_l-asl-en-how2sign",
        ]
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    # TODO: Check this
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config_yaml="config.yaml",
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
        return S2THubInterface(x["cfg"], x["task"], x["models"][0]) #Aquí s'utilitza, hauriem de mirar què necessitem i que es cadascun dels arguments d'aquestes.

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg):
        encoder = Sign2TextTransformerEncoder(cfg)
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
    def build_decoder(cls, cfg, task, embed_tokens):
        return TransformerDecoderScriptable(cfg, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance.""" #TODO: Check where the nones are coming from

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, cfg.decoder_embed_dim
        )
        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task, decoder_embed_tokens)
        return cls(encoder, decoder)

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

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class Sign2TextTransformerEncoder(FairseqEncoder):
    """Sign-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, cfg):
        super().__init__(None)

        self.encoder_freezing_updates = cfg.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=cfg.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(cfg.encoder_embed_dim)
        if cfg.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = Conv1dSubsampler(
            get_num_feats(
                SignFeatsType[cfg.feats_type],
                cfg.body_parts.split(','),
                cfg.feat_dims.split(',')
            ),
            cfg.conv_channels,
            cfg.encoder_embed_dim,
            [int(k) for k in cfg.conv_kernel_sizes.split(",")],
            [int(k) for k in cfg.conv_strides.split(",")],
        )

        self.embed_positions = PositionalEmbedding(
            cfg.max_source_positions, cfg.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        )
        if cfg.encoder_normalize_before:
            self.layer_norm = LayerNorm(cfg.encoder_embed_dim)
        else:
            self.layer_norm = None

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x) #changes the number of nans?

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
            "src_lengths": [],
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


class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None
