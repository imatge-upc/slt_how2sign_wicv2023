#!/usr/bin/env python3

import logging
import inspect
import math

from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .SL_perceiverIO_preprocessing import PerceiverImagePreprocessor
from .SL_perceiverIO_preprocessing import build_position_encoding

from fairseq import checkpoint_utils, utils
from fairseq.data.sign_language.utils import get_num_feats
# TODO: see how to use S2THubInterface
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    FairseqEncoder,
    FairseqDecoder,
)
from fairseq.models.transformer import Embedding

from dataclasses import dataclass, field
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from omegaconf import II

from fairseq.data.sign_language import SignFeatsType
logger = logging.getLogger(__name__)


@dataclass
class SLTopicDetectionPerceiverConfig(FairseqDataclass):
    '''
    Add model-specific arguments to the parser.
    '''
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability"}
    )
    encoder_input_embed_size: int = field(
        default=256, metadata={"help": "Used when features type is text or spot_align. Size of the input embedding"}
    )
    num_latents: int = field(
        default=256, metadata={'help': 'number of latent arrays'}
    )
    d_latents: int = field(
        default=1280, metadata={'help': 'number of channels in latent arrays'}
    )
    d_model: Optional[int] = field(
        default=None, metadata={'help': 'Dimension of the inputs. Should only be provided in case [*PerceiverTextPreprocessor*] is used or no preprocessor is provided.'}
    )
    num_blocks: int = field(
        default=1, metadata={'help': 'Number of blocks in the Transformer encoder'}
    )
    num_self_attends_per_block: int = field(
        default=4, metadata={'help': 'Number of self-attention layers per block'}
    )
    output_attentions: bool = field(
        default=False, metadata={'help': 'Wether to output attentions'}
    )
    output_hidden_states: bool = field(
        default=False, metadata={'help': 'Wether to output hidden states'}
    )
    num_self_attention_heads: int = field(
        default=8, metadata={'help': 'Number of attention heads for each self-attention layer in the Transformer encoder'}
    )
    num_cross_attention_heads: int = field(
        default=8, metadata={'help': 'number of heads in cross-att'}
    )
    chunk_size_feed_forward: int = field(
        default=768, metadata={'help': 'Chunk size for encoder\'s apply_chunking_to_forward'}
    )
    qk_channels: Optional[int] = field(
        default=None, metadata={'help': ('Dimension to project the queries + keys before applying attention'
                                         'in the cross-attention and self-attention layers of the encoder.'
                                         'Will default to preserving the dimension of the queries if not specified.')}
    )
    decoder_qk_channels: Optional[int] = field(
        default=None, metadata={'help': ('Dimension to project the queries + keys before applying attention'
                                         'in the cross-attention of the decoder.'
                                         'Will default to preserving the dimension of the queries if not specified.')}
    )
    v_channels: Optional[int] = field(
        default=768, metadata={'help': ('Dimension to project the values before applying attention'
                                         'in the cross-attention and self-attention layers of the encoder.'
                                         'Will default to preserving the dimension of the queries if not specified.')}
    )
    decoder_v_channels: Optional[int] = field(
        default=None, metadata={'help': ('Dimension to project the values before applying attention'
                                         'in the cross-attention of the decoder.'
                                         'Will default to preserving the dimension of the queries if not specified.')}
    )
    cross_attention_shape_for_attention: str = field(
        default='kv', metadata={'help': ('Dimension to use when downsampling the queries and keys in the cross-attention layer of the encoder.'
                                         'Possible values are "kv" and "q"')}
    )
    self_attention_widening_factor: int = field(
        default=1, metadata={'help': 'Dimension of the feed-forward layer in the cross-attention layer of the Transformer encoder.'}
    )
    cross_attention_widening_factor: int = field(
        default=1, metadata={'help': 'widening factor in cross-att MLP'}
    )
    hidden_act: str = field(
        default='gelu', metadata={'help': 'The non-linear activation function in the encoder and pooler.'}
    )
    attention_probs_dropout_prob: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    use_query_residual: bool = field(
        default=True, metadata={'help': 'Whether to add a query residual in the cross-attention layer of the encoder.'}
    )
    decoder_concat_preprocessed_input: bool = field(
        default=False, metadata={'help': 'Whether to concat inputs to the output query array.'}
    )
    num_bands: int = field(
        default=6, metadata={'help': 'Number of frequency bands in Fourier position encodings'}
    )
    preprocessor_position_encoding_type: str = field(
        default='fourier', metadata={'help': ('Position encoding type for PerceiverImagePreprocessor'
                                              'Can be "fourier" or "trainable"')}
    )
    decoder_position_encoding_type: str = field(
        default='fourier', metadata={'help': ('Position encoding type for PerceiverClassificationDecoder'
                                              'Can be "fourier" or "trainable"')}
    )
    image_prep_num_channels: int = field(
        default=256, metadata={'help': 'Number of channels in positional encoding of type Fourier Features, in PerceiverImagePreprocessor'}
    )
    image_prep_type: str = field(
        default='patches', metadata={'help': 'Preprocessing type for PerceiverImagePreprocessor'}
    )
    image_prep_spatial_downsample: int = field(
        default=4, metadata={'help': 'Spatial downsampling factor for PerceiverImagePreprocessor'}
    )
    image_prep_temporal_downsample: int = field(
        default=1, metadata={'help': 'Temporal downsampling factor for PerceiverImagePreprocessor'}
    )
    image_prep_in_channels: int = field(
        default=3, metadata={'help': 'Number of channels in the input for PerceiverImagePreprocessor'}
    )
    image_prep_out_channels: int = field(
        default=128, metadata={'help': 'Number of channels in the output for PerceiverImagePreprocessor'}
    )
    conv_after_patching: bool = field(
        default=False, metadata={'help': 'Whether to apply a convolutional layer after patching in PerceiverImagePreprocessor'}
    )
    conv_after_patching_in_channels: int = field(
        default=54, metadata={'help': 'Number of channels in the input of the convolutional layer after patching in PerceiverImagePreprocessor'}
    )
    load_pretrained_encoder_from: str = field(
        default="relu", metadata={"help": "model to take encoder weights from (for initialization)"}
    )
    load_pretrained_decoder_from: str = field(
        default="relu", metadata={"help": "model to take decoder weights from (for initialization)"}
    )
    encoder_freezing_updates: int = field(
        default=0, metadata={"help": "freeze encoder for first N updates"}
    )
    modeling_task: str = II("task.modeling_task")
    num_labels: int = II('task.num_labels')
    feats_type: ChoiceEnum([x.name for x in SignFeatsType]) = II("task.feats_type")
    body_parts: str = II("task.body_parts")
    feat_dims: str = II("task.feat_dims")
    max_source_positions: int = II("task.max_source_positions")


@register_model(
    'SL_topic_detection_PerceiverIO',
    dataclass=SLTopicDetectionPerceiverConfig
)
class PerceiverModel(FairseqEncoderDecoderModel):

    @classmethod
    def hub_models(cls):
        base_url = "" # TODO: Set base URL to upload checkpoints
        model_ids = [
            'SL_topic_detection_perceiver_s-how2sign',
            'SL_topic_detection_perceiver_m-how2sign',
            'SL_topic_detection_perceiver_l-how2sign',
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
        return S2THubInterface(x['cfg'], x['task'], x['models'][0])  # Aquí s'utilitza, hauriem de mirar què necessitem i que es cadascun dels arguments d'aquestes.

    def __init__(
        self,
        cfg,
        encoder,
        embeddings,
        decoder=None,
        input_preprocessor=None,
        output_postprocessor=None,
    ):
        '''
        Parameters:
            cfg ([`PerceiverConfig`]): Model configuration class with all the parameters of the model.
            decoder: Optional decoder to use to decode the latent representation of the encoder
            input_preprocessor: Optional input preprocessor to use
            output_postprocessor: Optional output postprocessor to use
        '''
        super().__init__(encoder, decoder)

        self.cfg = cfg

        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.embeddings = embeddings
        self.encoder = encoder
        self.decoder = decoder

        self.encoder_freezing_updates = cfg.encoder_freezing_updates

    def build_encoder(cfg, input_preprocessor, encoder_embed_tokens):
        encoder = PerceiverEncoder(
                    cfg,
                    kv_dim=(input_preprocessor.num_channels
                            if input_preprocessor is not None else
                            cfg.d_model),
                    encoder_embed_tokens=encoder_embed_tokens
                )
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

    def build_decoder(cfg):
        if cfg.modeling_task == 'classification':
            decoder_kwargs = {
                'cfg': cfg,

                'trainable_position_encoding_kwargs': {
                    'index_dims': 1,  # for classification, one single embedding suffices
                    'num_channels': cfg.num_labels,
                },

                'fourier_position_encoding_kwargs': {
                    'num_bands': cfg.num_bands,  # TODO: check this number
                    'max_resolution': [10],  # TODO: check if here we should specify T x H x W dims, check also H2S videos' resolution
                    'concat_pos': True,
                    'sine_only': False,
                },

                'num_channels': cfg.num_labels,
                'qk_channels': cfg.decoder_qk_channels,
                'v_channels': cfg.decoder_v_channels,
                'n_heads': cfg.num_cross_attention_heads,
                'widening_factor': cfg.cross_attention_widening_factor,
                'use_query_residual': cfg.use_query_residual,
                'concat_preprocessed_input': cfg.decoder_concat_preprocessed_input,
                'final_project': True,
                'position_encoding_only': False,
                'position_encoding_type': cfg.decoder_position_encoding_type,
            }
            decoder = PerceiverClassificationDecoder(**decoder_kwargs)
        else:
            err += (f'No decoder is available yet for modeling task "{cfg.modeling_task}".')
            logger.error(err)
        pretraining_path = getattr(cfg, "load_pretrained_decoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained decoder from: {pretraining_path}")
        return decoder

    def build_input_preprocessor(cfg):
        input_preprocessor = None
        if cfg.feats_type in ['video']:
            position_encoding_kwargs = {
                'trainable_position_encoding_kwargs': {
                    'index_dims': cfg.max_source_positions,
                    'num_channels': cfg.image_prep_num_channels,
                },
                'fourier_position_encoding_kwargs': {
                    'num_bands': cfg.num_bands,  # TODO: check this number
                    'max_resolution': [cfg.max_source_positions] + list(cfg.d_model),  # TODO: check if here we should specify T x H x W dims, check also H2S videos' resolution
                    'concat_pos': True,
                    'sine_only': False,
                },
            }
            input_preprocessor = PerceiverImagePreprocessor(
                cfg,
                prep_type=cfg.image_prep_type,
                spatial_downsample=cfg.image_prep_spatial_downsample,
                temporal_downsample=cfg.image_prep_temporal_downsample,
                position_encoding_type=cfg.preprocessor_position_encoding_type,
                in_channels=3,
                out_channels=cfg.image_prep_out_channels,
                conv_after_patching=cfg.conv_after_patching,
                conv_after_patching_in_channels=cfg.conv_after_patching_in_channels,  # only relevant when conv_after_patching = True
                conv2d_use_batchnorm=True,
                concat_or_add_pos='concat',
                project_pos_dim=-1,
                **position_encoding_kwargs,
            )
        return input_preprocessor

    def build_output_postprocessor(cfg):
        logger.info('No modeling task available yet for which an output postprocessor is needed.')
        return None

    def build_latent_embeddings(cfg):
        return PerceiverEmbeddings(cfg)

    @classmethod
    def build_model(cls, cfg, task):
        '''
        Build a new model instance.
        '''
        print(cfg.keys(), flush=True)

        cfg.d_model = (cfg.encoder_input_embed_size
                       if SignFeatsType[cfg.feats_type] in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings] else
                       get_num_feats(
                            SignFeatsType[cfg.feats_type],
                            cfg.body_parts.split(','),
                            cfg.feat_dims.split(',')
                       )
        )
        cfg.v_channels, cfg.d_latents = [min(cfg.v_channels, cfg.d_latents)] * 2

        decoder = cls.build_decoder(cfg)
        input_preprocessor = cls.build_input_preprocessor(cfg)
        output_postprocessor = cls.build_output_postprocessor(cfg)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)
        encoder_embed_tokens = None
        if SignFeatsType[cfg.feats_type] in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings]:
            encoder_embed_tokens = build_embedding(
                task.source_dictionary, cfg.encoder_input_embed_size
            )
        encoder = cls.build_encoder(cfg, input_preprocessor, encoder_embed_tokens)
        embeddings = cls.build_latent_embeddings(cfg)
        return cls(cfg, encoder, embeddings, decoder, input_preprocessor, output_postprocessor)

    def invert_attention_mask(self, encoder_attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        '''
        Invert an attention mask (e.g., switches 0. and 1.).
        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.
        Returns:
            `torch.Tensor`: The inverted attention mask.
        '''
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)  # fp16 compatibility

        if dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif dtype in [torch.bfloat16, torch.float32]:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                f"{dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
            )
        return encoder_extended_attention_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def get_head_mask(
        self, head_mask: Optional[torch.Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> torch.Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def get_normalized_probs(
        self,
        net_output: Tuple[torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.decoder.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def _forward(
        self,
        inputs,
        input_lengths,
        attention_mask=None,
        subsampled_output_points=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.cfg.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.cfg.output_hidden_states
        )

        if self.encoder.encoder_embed_tokens:
            inputs = self.encoder.encoder_embed_tokens(inputs)

        if self.input_preprocessor is not None:
            init_input_shape = inputs.size()
            inputs, modality_sizes, inputs_without_pos = self.input_preprocessor(inputs, network_input_is_1d=True)

            batch_size, seq_length, _ = inputs.size()
            input_lengths = torch.round(
                + input_lengths
                * seq_length
                / init_input_shape[1]
            )
        else:
            batch_size, seq_length, _ = inputs.size()
            modality_sizes = None
            inputs_without_pos = None

        if self.cfg.feats_type in ['i3d', 'keypoints', 'rotational'] and inputs.size()[-1] != self.cfg.d_model:
            raise ValueError(
                f"Last dimension of the inputs: {inputs.size()[-1]} doesn't correspond to cfg.d_model: {self.cfg.d_model}."
                " Make sure to set cfg.d_model appropriately."
            )

        if attention_mask is None:
            if input_lengths is not None:
                attention_mask = ~lengths_to_padding_mask(input_lengths.long())
            else:
                attention_mask = torch.ones(((batch_size, seq_length)), device=inputs.device)

        # Make the attention mask broadcastable to [batch_size, n_heads, seq_length, seq_length]
        extended_attention_mask = self.invert_attention_mask(attention_mask, inputs.dtype)

        # Prepare head mask if needed
        # 1.0 in head_mask indicates we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [n_heads] or [num_blocks x n_heads]
        # and head_mask is converted to shape [num_blocks x batch x n_heads x N x N]
        head_mask = self.get_head_mask(head_mask, self.cfg.num_blocks * self.cfg.num_self_attends_per_block)

        embedding_output = self.embeddings(batch_size=batch_size)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=None,
            head_mask=head_mask,
            inputs=inputs,
            inputs_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]

        logits = None
        if self.decoder:
            if subsampled_output_points is not None:  # TODO: when using video data, see if it is a good idea to subsample points from the video
                                                      #       instead of passing the whole frames, or all of the frames
                output_modality_sizes = {
                    "audio": subsampled_output_points["audio"].shape[0],
                    "image": subsampled_output_points["image"].shape[0],
                    "label": 1,
                }
            else:
                output_modality_sizes = None

            decoder_query = self.decoder.decoder_query(
                inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_output_points
            )

            decoder_outputs = self.decoder(
                decoder_query,
                z=sequence_output,
                query_mask=extended_attention_mask,
                output_attentions=output_attentions,
            )
            logits = decoder_outputs['logits']

            # add cross-attentions of decoder
            if output_attentions and decoder_outputs['cross_attentions'] is not None:
                encoder_outputs = encoder_outputs + decoder_outputs['cross_attentions']

            if self.output_postprocessor:
                logits = self.output_postprocessor(logits, modality_sizes=output_modality_sizes)

        return logits

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False, output_attentions=None):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens, src_lengths, output_attentions=output_attentions,
                )
        else:
            x = self._forward(
                src_tokens, src_lengths, output_attentions=output_attentions,
            )
        return x

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class PerceiverEmbeddings(nn.Module):
    '''
    Builds the latent embeddings.
    '''
    def __init__(self, cfg):
        super().__init__()
        self.latents = nn.Parameter(
            torch.randn(
                cfg.num_latents,
                cfg.d_latents
            )
        )

    def forward(self, batch_size):
        return self.latents.expand(batch_size, -1, -1)


class PerceiverSelfAttention(nn.Module):
    '''
    Multi-headed {self, cross}-attention. Can used both for encoding and decoding.
    '''
    def __init__(
        self,
        cfg,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        n_heads=1,
        q_dim=None,
        kv_dim=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if qk_channels is None:
            qk_channels = q_dim
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % n_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by n_heads ({n_heads}).")
        if v_channels % n_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by n_heads ({n_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // n_heads
        self.v_channels_per_head = self.v_channels // n_heads

        self.layernorm1 = nn.LayerNorm(q_dim)
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Projection matrices
        self.query = nn.Linear(q_dim, qk_channels)
        self.key = nn.Linear(kv_dim, qk_channels)
        self.key.bias = torch.nn.Parameter(torch.unsqueeze(self.key.bias, 0))
        self.value = nn.Linear(kv_dim, v_channels)
        self.value.bias = torch.nn.Parameter(torch.unsqueeze(self.value.bias, 0))

        self.dropout = nn.Dropout(cfg.attention_probs_dropout_prob)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_normal_(self.key.bias)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_normal_(self.value.bias)

    def transpose_for_scores(self, x, channels_per_head):
        new_x_shape = x.size()[:-1] + (self.n_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
    ):
        hidden_states = self.layernorm1(hidden_states)
        inputs = self.layernorm2(inputs)

        # Project queries, keys and values to a common feature dimension. If this is instantiated as a cross-attention module,
        # the keys and values come from the inputs; the attention mask ensures that non-relevant tokens are not attended to.
        is_cross_attention = inputs is not None
        queries = self.query(hidden_states)

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
            attention_mask = inputs_mask
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        # Reshape channels for multi-head attention.
        # B x T x C --> B x n_heads x T x CperHead
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)

        # Take the dot product between the queries and keys to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        _, _, _, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hiddens = self.n_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)
        if attention_mask is not None:
            # Apply the attention mask
            attention_scores = attention_scores + attention_mask
        # Normalize the att scores to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Mask heads if desired
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class PerceiverSelfOutput(nn.Module):
    def __init__(self, cfg, in_channels, out_channels):
        super().__init__()
        self.dense = nn.Linear(in_channels, out_channels)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        return hidden_states


class PerceiverAttention(nn.Module):
    '''
    Attention module, including a dense block.
    '''
    def __init__(
        self,
        cfg,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        n_heads=1,
        q_dim=None,
        kv_dim=None,
        use_query_residual=True,
    ):
        super().__init__()
        # MultiHead attention
        if is_cross_attention and qk_channels is None:
            if cfg.cross_attention_shape_for_attention == 'q':
                qk_channels = q_dim
            elif cfg.cross_attention_shape_for_attention == 'kv':
                qk_channels = kv_dim
            else:
                raise ValueError(
                    f'Unknown value {cfg.cross_attention_shape_for_attention} for '
                     'cross_attention_shape_for_attention.'
                )
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels
        self.self = PerceiverSelfAttention(
            cfg,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            n_heads=n_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
        )
        # dense block
        out_channels = None
        if is_cross_attention:
            out_channels = q_dim
        else:
            if out_channels is None:
                out_channels = v_channels
        self.output = PerceiverSelfOutput(cfg, in_channels=self.self.v_channels, out_channels=out_channels)
        self.use_query_residual = use_query_residual

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )

        # Output projection
        attention_output = self.output(self_outputs[0])

        # Optionally include a residual to the original queries.
        if self.use_query_residual:
            attention_output = attention_output + hidden_states

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PerceiverMLP(nn.Module):
    '''
    A Transformer-style dense module to follow attention.
    '''
    def __init__(self, cfg, input_size, widening_factor):
        super().__init__()
        self.dense1 = nn.Linear(input_size, widening_factor * input_size)
        self.intermediate_act_fn = utils.get_activation_fn(cfg.hidden_act)
        self.dense2 = nn.Linear(widening_factor * input_size, input_size)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class PerceiverLayer(nn.Module):
    def __init__(
        self,
        cfg,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        n_heads=1,
        q_dim=None,
        kv_dim=None,
        widening_factor=4,
        use_query_residual=True,
    ):
        super().__init__()
        self.chunk_size_feed_forward = cfg.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PerceiverAttention(
            cfg,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            n_heads=n_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
        )
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(cfg, input_size=q_dim, widening_factor=widening_factor)

    def apply_chunking_to_forward(
        self, forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
    ) -> torch.Tensor:
        '''
        Credits to: https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/modeling_utils.py

        This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
        `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.
        If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
        applying `forward_fn` to `input_tensors`.
        Args:
            forward_fn (`Callable[..., torch.Tensor]`):
                The forward function of the model.
            chunk_size (`int`):
                The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
            chunk_dim (`int`):
                The dimension over which the `input_tensors` should be chunked.
            input_tensors (`Tuple[torch.Tensor]`):
                The input tensors of `forward_fn` which will be chunked
        Returns:
            `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.
        '''

        assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

        # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
        num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
        if num_args_in_forward_chunk_fn != len(input_tensors):
            raise ValueError(
                f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
                "tensors are given"
            )

        if chunk_size > 0:
            tensor_shape = input_tensors[0].shape[chunk_dim]
            for input_tensor in input_tensors:
                if input_tensor.shape[chunk_dim] != tensor_shape:
                    raise ValueError(
                        f"All input tenors have to be of the same shape: {tensor_shape}, "
                        f"found shape {input_tensor.shape[chunk_dim]}"
                    )

            if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
                raise ValueError(
                    f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                    f"size {chunk_size}"
                )

            num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

            # chunk input tensor into tuples
            input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
            # apply forward fn to every tuple
            output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
            # concatenate output at same dimension
            return torch.cat(output_chunks, dim=chunk_dim)

        return forward_fn(*input_tensors)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
    ):
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]

        chunk_size = min(attention_output.shape[self.seq_len_dim], self.chunk_size_feed_forward)
        layer_output = self.apply_chunking_to_forward(
            self.feed_forward_chunk, chunk_size, self.seq_len_dim, attention_output
        )

        layer_output = layer_output + attention_output  # residual connection

        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output attention weights
        return outputs

    def feed_forward_chunk(self, attention_output):
        layer_output = self.layernorm(attention_output)
        layer_output = self.mlp(layer_output)
        return layer_output


class PerceiverEncoder(FairseqEncoder):
    '''
    The Perceiver Encoder: a scalable, fully attentional encoder.
    '''
    def __init__(self, cfg, kv_dim=None, encoder_embed_tokens=None):
        super().__init__(cfg)
        self.cfg = cfg

        self.encoder_embed_tokens = encoder_embed_tokens

        # Make sure we can use multihead-attention with these shapes.
        if cfg.d_latents % cfg.num_self_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({cfg.d_latents}) must be divisible by"
                f" num_self_attend_heads ({cfg.num_self_attention_heads})."
            )
        if cfg.d_latents % cfg.num_cross_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({cfg.d_latents}) must be divisible by"
                f" num_cross_attend_heads ({cfg.num_cross_attention_heads})."
            )

        if kv_dim is None:
            kv_dim = (cfg.encoder_input_embed_size
                        if SignFeatsType[cfg.feats_type] in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings] else
                        get_num_feats(
                                SignFeatsType[cfg.feats_type],
                                cfg.body_parts.split(','),
                                cfg.feat_dims.split(',')
                        )
                    )
        # Construct the cross attention layer.
        self.cross_attention = PerceiverLayer(
            cfg,
            is_cross_attention=True,
            qk_channels=cfg.qk_channels,
            v_channels=cfg.v_channels,
            n_heads=cfg.num_cross_attention_heads,
            q_dim=cfg.d_latents,
            kv_dim=kv_dim,
            widening_factor=cfg.cross_attention_widening_factor,
            use_query_residual=cfg.use_query_residual,
        )

        # Construct a single block of self-attention layers.
        # We get deeper architectures by applying this block more than once.
        self_attention_layers = []
        for _ in range(cfg.num_self_attends_per_block):
            layer = PerceiverLayer(
                cfg,
                is_cross_attention=False,
                qk_channels=cfg.qk_channels,
                v_channels=cfg.v_channels,
                n_heads=cfg.num_self_attention_heads,
                q_dim=cfg.d_latents,
                kv_dim=cfg.d_latents,
                widening_factor=cfg.self_attention_widening_factor,
            )
            self_attention_layers.append(layer)

        self.self_attends = nn.ModuleList(self_attention_layers)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        # Apply the cross-attention between the latents (hidden_states) and inputs:
        layer_outputs = self.cross_attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            inputs=inputs,
            inputs_mask=inputs_mask,
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_cross_attentions = all_cross_attentions + (layer_outputs[1],)

        # Apply the block of self-attention layers more than once:
        for _ in range(self.cfg.num_blocks):
            for i, layer_module in enumerate(self.self_attends):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # return tuple(
        #     v
        #     for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
        #     if v is not None
        # )
        return (hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions)


# Below: IO pre- and post-processor classes for Perceiver.

class PerceiverBasicDecoder(nn.Module):
    """
    Cross-attention-based decoder. This class can be used to decode the final hidden states of the latents using a
    cross-attention operation, in which the latents produce keys and values.
    The shape of the output of this class depends on how one defines the output queries (also called decoder queries).
    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        output_num_channels (`int`, *optional*):
            The number of channels in the output. Will only be used in case *final_project* is set to `True`.
        position_encoding_type (`str`, *optional*, defaults to "trainable"):
            The type of position encoding to use. Can be either "trainable", "fourier", or "none".
        output_index_dims (`int`, *optional*):
            The number of dimensions of the output queries. Ignored if 'position_encoding_type' == 'none'.
        num_channels (`int`, *optional*):
            The number of channels of the decoder queries. Ignored if 'position_encoding_type' == 'none'.
        qk_channels (`int`, *optional*):
            The number of channels of the queries and keys in the cross-attention layer.
        v_channels (`int`, *optional*, defaults to 128):
            The number of channels of the values in the cross-attention layer.
        n_heads (`int`, *optional*, defaults to 1):
            The number of attention heads in the cross-attention layer.
        widening_factor (`int`, *optional*, defaults to 1):
            The widening factor of the cross-attention layer.
        use_query_residual (`bool`, *optional*, defaults to `False`):
            Whether to use a residual connection between the query and the output of the cross-attention layer.
        concat_preprocessed_input (`bool`, *optional*, defaults to `False`):
            Whether to concatenate the preprocessed input to the query.
        final_project (`bool`, *optional*, defaults to `True`):
            Whether to project the output of the cross-attention layer to a target dimension.
        position_encoding_only (`bool`, *optional*, defaults to `False`):
            Whether to only use this class to define output queries.
    """

    def __init__(
        self,
        cfg,
        output_num_channels,
        position_encoding_type="trainable",
        # The following 2 arguments are ignored if position_encoding_type == 'none':
        output_index_dims=None,
        num_channels=128,
        subsampled_index_dims=None,
        qk_channels=None,
        v_channels=None,
        n_heads=1,
        widening_factor=1,
        use_query_residual=False,
        concat_preprocessed_input=False,
        final_project=True,
        position_encoding_only=False,
        **position_encoding_kwargs,
    ):
        super().__init__()

        self.output_num_channels = output_num_channels
        # If `none`, the decoder will not construct any position encodings.
        # You should construct your own when quering the decoder.
        self.output_position_encodings = None
        self.position_encoding_type = position_encoding_type
        self.position_encoding_kwargs = position_encoding_kwargs

        if position_encoding_type != "none":
            self.output_position_encodings, self.positions_projection = build_position_encoding(
                position_encoding_type=position_encoding_type, **position_encoding_kwargs
            )

        self.output_index_dims = output_index_dims
        self.num_channels = num_channels
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self.subsampled_index_dims = subsampled_index_dims
        self.concat_preprocessed_input = concat_preprocessed_input
        self.final_project = final_project
        self.position_encoding_only = position_encoding_only

        # for multimodal autoencoding, we don't need the decoder cross-attention and final layer
        # so then we will set position_encoding_only to True
        if not self.position_encoding_only:
            self.decoding_cross_attention = PerceiverLayer(
                cfg,
                is_cross_attention=True,
                qk_channels=qk_channels,
                v_channels=v_channels,
                n_heads=n_heads,
                q_dim=num_channels,
                kv_dim=cfg.d_latents,
                widening_factor=widening_factor,
                use_query_residual=use_query_residual,
            )
            self.final_layer = nn.Linear(num_channels, output_num_channels) if final_project else nn.Identity()

    @property
    def num_query_channels(self) -> int:
        if self.position_encoding_type == "none":  # Queries come from elsewhere
            raise ValueError(
                "You cannot calculate number of decoder query channels when position_encoding_type is set to none"
            )
        if self.position_encoding_only:
            if "project_pos_dim" in self.position_encoding_kwargs:
                return self.position_encoding_kwargs["project_pos_dim"]
            return self.output_position_encodings.output_size()
        if self.final_project:
            return self.output_num_channels
        return self.num_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        if self.position_encoding_type == "none":  # Queries come from elsewhere
            raise ValueError("You cannot construct decoder queries when position_encoding_type is set to none")
        if subsampled_points is not None:
            # subsampled_points are the indices if the inputs would be flattened
            # however, the inputs aren't flattened, that's why we use unravel_index
            # to get the indices for the unflattened array
            # unravel_index returns a tuple (x_idx, y_idx, ...)
            # stack to get the [n, d] tensor of coordinates
            indices = list(
                torch.from_numpy(x) for x in np.unravel_index(subsampled_points.cpu(), self.output_index_dims)
            )
            pos = torch.stack(indices, dim=1)
            batch_size = inputs.shape[0]
            # Map these coordinates to [-1, 1]
            pos = -1 + 2 * pos / torch.tensor(self.output_index_dims)[None, :]
            pos = torch.broadcast_to(pos[None], [batch_size, pos.shape[0], pos.shape[1]])
            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(
                    self.output_index_dims, batch_size=batch_size, device=inputs.device, pos=pos
                )

            # Optionally project them to a target dimension.
            pos_emb = self.positions_projection(pos_emb)
            pos_emb = torch.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            batch_size = inputs.shape[0]
            index_dims = inputs.shape[2:]

            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(index_dims, batch_size, device=inputs.device)

            # Optionally project them to a target dimension.
            pos_emb = self.positions_projection(pos_emb)

        if self.concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError("Value is required for inputs_without_pos if concat_preprocessed_input is True")
            pos_emb = torch.cat([inputs_without_pos, pos_emb], div=-1)

        return pos_emb

    def forward(self, query, z, query_mask=None, output_attentions=False):
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        cross_attentions = () if output_attentions else None
        layer_outputs = self.decoding_cross_attention(
            query,
            attention_mask=query_mask,
            head_mask=None,
            inputs=z,
            inputs_mask=None,
            output_attentions=output_attentions,
        )
        output = layer_outputs[0]

        if output_attentions:
            cross_attentions = cross_attentions + (layer_outputs[1],)

        logits = self.final_layer(output)

        # return PerceiverDecoderOutput(logits=logits, cross_attentions=cross_attentions)
        return {
            'logits': logits,
            'cross_attentions': cross_attentions
        }


class PerceiverClassificationDecoder(FairseqDecoder):
    '''
    Cross-attention based classification decoder. Light-weight wrapper of [`PerceiverBasicDecoder`] for logit output.
    Will turn the output of the Perceiver encoder which is of shape (batch_size, num_latents, d_latents) to a tensor of
    shape (batch_size, num_labels). The queries are of shape (batch_size, 1, num_labels).
    Args:
        config:
            Model configuration.
    '''
    def __init__(self, cfg, **decoder_kwargs):
        super().__init__(cfg)


        self.num_labels = cfg.num_labels
        self.decoder = PerceiverBasicDecoder(
            cfg,
            output_num_channels=self.num_labels,
            output_index_dims=1,  # Predict a single logit array.
            **decoder_kwargs,
        )

    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ):
        '''
        Get normalized probabilities (or log probs) from a net's output.
        '''

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return self.decoder.decoder_query(
            inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_points
        )

    def forward(self, query, z, query_mask=None, output_attentions=False):
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)

        # B x 1 x num_classes -> B x num_classes
        logits = decoder_outputs['logits'][:, 0, :]

        return {
            'logits': logits,
            'cross_attentions' : decoder_outputs['cross_attentions'],
        }
