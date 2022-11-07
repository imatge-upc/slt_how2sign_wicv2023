import logging
logger = logging.getLogger(__name__)

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.models.speech_to_text.hub_interface import S2THubInterface  # TODO: això haurem de veure en què s'utilitza i com ho fem nosaltres, si utilitzem el mateix.
from fairseq import checkpoint_utils
from fairseq.data.sign_language.utils import get_num_feats
from fairseq.models import (
    BaseFairseqModel,
    register_model,
)
from fairseq.models.transformer import Embedding
from fairseq.models.lstm import LSTMEncoder, LSTM
from fairseq.modules import (
    FairseqDropout,
)

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import ChoiceEnum
from fairseq.data.sign_language import SignFeatsType

from omegaconf import II


@dataclass
class SLTopicDetectionLSTMConfig(FairseqDataclass):
    """Add model-specific arguments to the parser."""
    # input
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability"}
    )
    encoder_input_embed_size: int = field(
        default=256, metadata={"help": "Used when features type is text or spot_align or mouthings. Size of the input embedding"}
    )
    encoder_hidden_size: int = field(
        default=256, metadata={"help": "encoder hidden dimension"}
    )
    encoder_bidirectional: bool = field(
        default=True, metadata={"help": 'make all layers of encoder bidirectional'}
    )
    encoder_cells: int = field(
        default=1, metadata={"help": "num encoder cells"}
    )
    encoder_hid_attention: bool = field(
        default=False, metadata={'help': 'if True, use attention over encoder hidden states'}
    )
    encoder_input_attention: bool = field(
        default=False, metadata={'help': 'if True, use attention over input token and previous hidden state'}
    )
    att_size: int = field(
        default=256, metadata={'help': 'size of attention network for input attention'}
    )
    load_pretrained_encoder_from: str = field(
        default="relu", metadata={"help": "model to take encoder weights from (for initialization)"}
    )
    load_pretrained_classif_head_from: str = field(
        default="relu", metadata={"help": "model to take classification head weights from (for initialization)"}
    )
    encoder_freezing_updates: int = field(
        default=0, metadata={"help": "freeze encoder for first N updates"}
    )
    pad: int = field(
        default=1, metadata={'help': 'index along which to pad sequences'}
    )
    left_pad: bool = field(
        default=True, metadata={'help': 'if True, pad to the left'}
    )

    feats_type: ChoiceEnum([x.name for x in SignFeatsType]) = II("task.feats_type")
    body_parts: str = II("task.body_parts")
    feat_dims: str = II("task.feat_dims")
    max_source_positions: int = II("task.max_source_positions")


@register_model("SL_topic_detection_LSTM", dataclass=SLTopicDetectionLSTMConfig)
class SLTopicDetectionLSTMModel(BaseFairseqModel):
    '''
    LSTM model that receives as input a sequence of 1D tensors
    '''

    @classmethod
    def hub_models(cls):
        base_url = "" # TODO: Set base URL to upload checkpoints
        model_ids = [
            'SL_topic_detection_LSTM-asl-en-how2sign',
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
        return S2THubInterface(x["cfg"], x["task"], x["models"][0])
        # Aquí s'utilitza, hauriem de mirar què necessitem i què és cadascun dels arguments d'aquests.

    def __init__(self, cfg, encoder, att_encoder, classif_head):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.att_encoder = att_encoder
        self.classif_head = classif_head

    @classmethod
    def build_encoder(cls, cfg, encoder_embed_tokens=None):
        if cfg.encoder_input_attention:
            encoder = SLTopicDetectionLSTMEncoderAttention(cfg)
        else:
            encoder = SLTopicDetectionLSTMEncoder(cfg, encoder_embed_tokens)
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
        classif_head = SLTopicDetectionLSTMClassifHead(cfg)
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
        '''Build a new model instance.'''
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        encoder_embed_tokens = None
        if SignFeatsType[cfg.feats_type] in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings]:
            encoder_embed_tokens = build_embedding(
                task.source_dictionary, cfg.encoder_input_embed_size
            )
        encoder = cls.build_encoder(cfg, encoder_embed_tokens)

        att_encoder = None

        print(cfg.keys(), flush=True)

        if cfg.encoder_hid_attention:
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
        if self.cfg.encoder_hid_attention:
            x = self.att_encoder(x['encoder_out'][0])[0]  # ignore attention weights and keep embeddings
        else:
            x = x['encoder_out'][0][-1,:,:]

        x = self.classif_head(x)
        return x


class SLTopicDetectionLSTMEncoder(LSTMEncoder):
    '''
    LSTM encoder.
    '''

    def __init__(
        self,
        cfg,
        encoder_embed_tokens=None
    ):
        super().__init__(cfg, num_layers=cfg.encoder_cells, padding_idx=cfg.pad, pretrained_embed='mock_embed')
        self.cfg = cfg
        self.num_layers = cfg.encoder_cells
        self.encoder_embed_tokens = encoder_embed_tokens
        self.dropout_out_module = FairseqDropout(
            cfg.dropout * 1.0, module_name=self.__class__.__name__
        )
        self.bidirectional = cfg.encoder_bidirectional
        self.hidden_size = cfg.encoder_hidden_size
        self.max_source_positions = cfg.max_source_positions

        self.lstm = LSTM(
            input_size=(cfg.encoder_input_embed_size
                        if SignFeatsType[cfg.feats_type] in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings] else
                        get_num_feats(
                                SignFeatsType[cfg.feats_type],
                                cfg.body_parts.split(','),
                                cfg.feat_dims.split(',')
                        )
            ),
            hidden_size=cfg.encoder_hidden_size,
            num_layers=cfg.encoder_cells,
            dropout=self.dropout_out_module.p if cfg.encoder_cells > 1 else 0.0,
            bidirectional=cfg.encoder_bidirectional,
        )
        self.left_pad = cfg.left_pad

        self.output_units = cfg.encoder_hidden_size
        if cfg.encoder_bidirectional:
            self.output_units *= 2

        self.encoder_freezing_updates = cfg.encoder_freezing_updates
        self.num_updates = 0

        self.padding_idx = 1

    def _forward(
            self,
            src_tokens: Tensor,
            src_lengths: Tensor,
            enforce_sorted: bool = True,
        ):
            '''
            Args:
                src_tokens (LongTensor): tokens in the source language of
                    shape `(batch, src_len)`
                src_lengths (LongTensor): lengths of each source sentence of
                    shape `(batch)`
                enforce_sorted (bool, optional): if True, `src_tokens` is
                    expected to contain sequences sorted by length in a
                    decreasing order. If False, this condition is not
                    required. Default: True.
            '''
            x = src_tokens
            if self.encoder_embed_tokens:
                x = self.encoder_embed_tokens(x)
            bsz, input_lengths, _ = x.size()
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            # pack embedded source tokens into a PackedSequence
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, src_lengths.cpu(), enforce_sorted=enforce_sorted
            )

            # apply LSTM
            if self.bidirectional:
                state_size = 2 * self.num_layers, bsz, self.hidden_size
            else:
                state_size = self.num_layers, bsz, self.hidden_size
            h0 = x.new_zeros(*state_size)
            c0 = x.new_zeros(*state_size)
            packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

            # unpack outputs and apply dropout
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_outs, padding_value=self.padding_idx * 1.0
            )
            x = self.dropout_out_module(x)
            assert list(x.size()) == [input_lengths, bsz, self.output_units]

            if self.bidirectional:
                final_hiddens = self.combine_bidir(final_hiddens, bsz)
                final_cells = self.combine_bidir(final_cells, bsz)

            # encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
            encoder_padding_mask = src_tokens.eq(self.padding_idx).transpose(0, 1)

            return {
                    'encoder_out': [x],  # seq_len x batch x num_directions*hidden
                    'final_hiddens': final_hiddens,  # num_layers x batch x num_directions*hidden
                    'final_cells': final_cells,  # num_layers x batch x num_directions*hidden
                    'encoder_padding_mask': [encoder_padding_mask],  # seq_len x batch
                    'src_tokens': [],
                    'src_lengths': [src_lengths],
            }

    def forward(self, src_tokens, src_lengths):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                        src_tokens, src_lengths, enforce_sorted='albert'!=SignFeatsType[self.cfg.feats_type].name[-6:]
                )
        else:
            x = self._forward(
                src_tokens, src_lengths, enforce_sorted='albert'!=SignFeatsType[self.cfg.feats_type].name[-6:]
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

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            'final_hiddens': encoder_out['final_hiddens'],  # num_layers x batch x num_directions*hidden
            'final_cells': encoder_out['final_cells'],  # num_layers x batch x num_directions*hidden
            "src_tokens": [],  # B x T
            "src_lengths": encoder_out['src_lengths'],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class SLTopicDetectionLSTMEncoderAttention(SLTopicDetectionLSTMEncoder):
    '''
    LSTM encoder with attention over inputs.
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
    '''

    def __init__(self, cfg):
        super().__init__(cfg)
        self.input_size=(cfg.encoder_input_embed_size
                        if SignFeatsType[cfg.feats_type] in [SignFeatsType.text, SignFeatsType.spot_align, SignFeatsType.mouthings] else
                        get_num_feats(
                                SignFeatsType[cfg.feats_type],
                                cfg.body_parts.split(','),
                                cfg.feat_dims.split(',')
                        )
        )
        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(
                nn.LSTMCell(input_size=self.input_size,
                            hidden_size=self.hidden_size)
            )
            if self.bidirectional:
                self.b_cells.append(
                    nn.LSTMCell(input_size=self.input_size,
                                hidden_size=self.hidden_size)
                )
            self.input_size = self.hidden_size
        self.input_att = InputAttentionLayer(cfg)

    def _forward(
            self,
            src_tokens: Tensor,
            src_lengths: Tensor,
        ):
            '''
            Args:
                src_tokens (LongTensor): tokens in the source language of
                    shape `(batch, src_len)`
                src_lengths (LongTensor): lengths of each source sentence of
                    shape `(batch)`
                enforce_sorted (bool, optional): if True, `src_tokens` is
                    expected to contain sequences sorted by length in a
                    decreasing order. If False, this condition is not
                    required. Default: True.
            '''

            bsz, input_lengths, _ = src_tokens.size()
            # B x T x C -> T x B x C
            x = src_tokens.transpose(0, 1)

            # apply LSTM
            state_size = bsz, self.hidden_size
            h_f = [x.new_zeros(*state_size)] * self.num_layers
            c_f = [x.new_zeros(*state_size)] * self.num_layers
            if self.bidirectional:
                h_b = [x.new_zeros(*state_size)] * self.num_layers
                c_b = [x.new_zeros(*state_size)] * self.num_layers
            out = []

            for t in range(input_lengths):
                in_f = self.input_att(x[t, :, :], h_f[-1])
                for i in range(self.num_layers):
                    h_f[i], c_f[i] = self.f_cells[i](in_f, (h_f[i], c_f[i]))
                    in_f = h_f[i]
                out.append(h_f[-1])

                if self.bidirectional:
                    in_b = self.input_att(x[-(t+1), :, :], h_b[-1])
                    for i in range(self.num_layers):
                        h_b[i], c_b[i] = self.b_cells[i](in_b, (h_b[i], c_b[i]))
                        in_b = h_b[i]
                    out[t] = torch.cat( (out[t], h_b[-1]), 1 )

            out = torch.stack(out)
            out = self.dropout_out_module(out)
            assert list(out.size()) == [input_lengths, bsz, self.output_units]

            return {
                    'encoder_out': [out],  # seq_len x batch x num_directions*hidden
            }


class SLTopicDetectionLSTMClassifHead(BaseFairseqModel):
    '''
    Classification head
    '''
    def __init__(self, cfg):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(
                cfg.encoder_hidden_size * (1 + cfg.encoder_bidirectional),
                cfg.encoder_hidden_size * (1 + cfg.encoder_bidirectional) // 2,
            ),
            nn.ReLU(),
            nn.Linear(cfg.encoder_hidden_size * (1 + cfg.encoder_bidirectional) // 2, 10),
        )

    def forward(self, x):
        # x: batch x hidden
        return self.classifier(x)


class EncoderAttentionLayer(BaseFairseqModel):
    '''
    Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
    https://aclanthology.org/P16-2034/
    '''
    def __init__(self, cfg):
        super().__init__()
        self.w = torch.nn.parameter.Parameter(
            data=torch.randn(
                    cfg.encoder_hidden_size * (1 + cfg.encoder_bidirectional)
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


class InputAttentionLayer(nn.Module):
    '''
    Attention Layer.
    Attends over input and previous hidden state, to compute weight matrix for input.
    '''

    def __init__(self, cfg):
        super().__init__()
        self.in_size = get_num_feats(
                            SignFeatsType[cfg.feats_type],
                            cfg.body_parts.split(','),
                            cfg.feat_dims.split(',')
                        )
        self.input_att = nn.Linear(  # linear layer to transform input features
            self.in_size,
            cfg.att_size
        )
        self.hid_att = nn.Linear(cfg.encoder_hidden_size, cfg.att_size)  # linear layer to transform hidden state
        self.full_att = nn.Linear(cfg.att_size, self.in_size)  # linear layer to calculate values to be softmax-ed

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        '''
        Forward propagation.
        :param encoder_out: input samples, a tensor of dimension (batch_size, input_dim)
        :param hidden: previous hidden state, a tensor of dimension (batch_size, cfg.encoder_hidden_size)
        :return: attention weighted encoding, weights
        '''
        att1 = self.input_att(x)  # batch x att_size
        att2 = self.hid_att(hidden)  # batch x att_size
        att = self.full_att(self.tanh(att1 + att2))  # batch x encoder_hidden_size
        alpha = self.softmax(att)
        attention_weighted_x = x * alpha  # batch x self.in_size
        return attention_weighted_x
