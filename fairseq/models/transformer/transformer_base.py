# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import random
import numpy as np
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerEncoderBase,
    TransformerDecoderBase,
    TransformerConfig,
)
from torch import Tensor
from embedding import MorphTEmbedding, MorphLSTMEmbedding, NNEmbedding
import json


class TransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )

            src_mode = cfg.emb_mode.lower()

            if src_mode == 'MorphTE'.lower():
                num_morphemes, num_words, morph_index_matrix = cls.get_morphInfo_jointed(src_dict, cfg.mor_path)
                encoder_embed_tokens = MorphTEmbedding(num_morphemes, num_words, cfg.encoder.embed_dim,
                                                       morph_index_matrix, rank=cfg.emb_rank, padding_idx=src_dict.pad())

            elif src_mode == 'MorphLSTM'.lower():
                num_morphemes, num_words, morph_index_matrix = cls.get_morphInfo_jointed(src_dict, cfg.mor_path)
                encoder_embed_tokens = MorphLSTMEmbedding(num_morphemes, num_words, cfg.encoder.embed_dim,
                                                          morph_index_matrix, padding_idx=src_dict.pad())

            elif src_mode == 'Original'.lower():
                encoder_embed_tokens = NNEmbedding(len(src_dict), cfg.encoder.embed_dim, padding_idx=src_dict.pad())

            else:
                raise NotImplementedError

            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True

        else:
            src_mode = cfg.emb_mode.lower()
            tgt_mode = src_mode

            if src_mode == 'MorphTE'.lower():
                 num_morphemes, num_words, morph_index_matrix = cls.get_morphInfo(src_dict, lang='src', mor_path=cfg.mor_path)
                 encoder_embed_tokens = MorphTEmbedding(num_morphemes, num_words, cfg.encoder.embed_dim,
                                                        morph_index_matrix, rank=cfg.emb_rank,
                                                        padding_idx=src_dict.pad())

            elif src_mode == 'MorphLSTM'.lower():
                num_morphemes, num_words, morph_index_matrix = cls.get_morphInfo(src_dict, lang='src', mor_path=cfg.mor_path)
                encoder_embed_tokens = MorphLSTMEmbedding(num_morphemes, num_words, cfg.encoder.embed_dim,
                                                          morph_index_matrix, padding_idx=src_dict.pad())

            elif src_mode == 'Original'.lower():
                encoder_embed_tokens = NNEmbedding(len(src_dict), cfg.encoder.embed_dim, padding_idx=src_dict.pad())

            else:
                raise NotImplementedError


            if tgt_mode == 'MorphTE'.lower():
                num_morphemes, num_words, morph_index_matrix = cls.get_morphInfo(tgt_dict, lang='tgt',
                                                                                 mor_path=cfg.mor_path)
                decoder_embed_tokens = MorphTEmbedding(num_morphemes, num_words, cfg.decoder.embed_dim,
                                                       morph_index_matrix, rank=cfg.emb_rank,
                                                       padding_idx=tgt_dict.pad())

            elif tgt_mode == 'MorphLSTM'.lower():
                num_morphemes, num_words, morph_index_matrix = cls.get_morphInfo(tgt_dict, lang='tgt',
                                                                                 mor_path=cfg.mor_path)
                decoder_embed_tokens = MorphLSTMEmbedding(num_morphemes, num_words, cfg.decoder.embed_dim,
                                                          morph_index_matrix, padding_idx=tgt_dict.pad())

            elif tgt_mode == 'Original'.lower():
                decoder_embed_tokens = NNEmbedding(len(tgt_dict), cfg.decoder.embed_dim, padding_idx=tgt_dict.pad())

            else:
                raise NotImplementedError

        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        if not cfg.share_all_embeddings:
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=cfg.min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=cfg.min_params_to_wrap)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb


    @classmethod
    def get_morphInfo_jointed(cls, dictionary, mor_path=None):

        word2id = dictionary.indices

        word2mor = json.load(open(mor_path))['word2mor']

        add_symbols = list((set(word2id.keys()) - set(word2mor.keys())))
        if '<pad>' in add_symbols:
            add_symbols.remove('<pad>')

        add_symbols = ['<pad>', '[mor-PAD-1]', '[mor-PAD-2]', '[mor-UNK]'] + add_symbols

        mor_set = add_symbols + json.load(open(mor_path))['mor_set']

        mor2id = {}
        for i, mor in enumerate(mor_set):
            mor2id[mor] = i

        word2morid = {}
        for k, v in word2mor.items():
            mor_li = v
            if len(mor_li) > 3:
                mor_li = mor_li[:3]

            if len(mor_li) == 1:
                mor_li = mor_li + ['[mor-PAD-1]', '[mor-PAD-2]']

            if len(mor_li) == 2:
                mor_li = mor_li + ['[mor-PAD-2]']

            assert len(mor_li) == 3
            word2morid[k] = [mor2id.get(mor, mor2id['[mor-UNK]']) for mor in mor_li]

        for w in (set(word2id.keys()) - set(word2mor.keys())):
            word2morid[w] = [mor2id[w]] + [mor2id['[mor-PAD-1]'], mor2id['[mor-PAD-2]']]

        co_matrix = []
        for kv in sorted(word2id.items(), key=lambda kv: (kv[1], kv[0])):
            co_matrix.append(word2morid.get(kv[0]))
        co_matrix = torch.LongTensor(co_matrix)      # Morpheme Index Matrix

        print("********************", "jointed dict")
        print("********************", "num_morphemes", len(mor2id), "num_words", len(word2id))
        # print("********************", set(word2id.keys()) - set(word2morid.keys()))

        return len(mor2id), len(word2id), co_matrix


    @classmethod
    def get_morphInfo(cls, dictionary, lang='src', mor_path=None):

        word2id = dictionary.indices

        if 'src' in lang:
            word2mor = json.load(open(mor_path))['word2mor_src']
        else:
            word2mor = json.load(open(mor_path))['word2mor_tgt']

        add_symbols = list((set(word2id.keys()) - set(word2mor.keys())))
        if '<pad>' in add_symbols:
            add_symbols.remove('<pad>')

        add_symbols = ['<pad>', '[mor-PAD-1]', '[mor-PAD-2]', '[mor-UNK]'] + add_symbols

        if 'src' in lang:
            mor_set = add_symbols + json.load(open(mor_path))['mor_set_src']
        else:
            mor_set = add_symbols + json.load(open(mor_path))['mor_set_tgt']

        mor2id = {}
        for i, mor in enumerate(mor_set):
            mor2id[mor] = i

        word2morid = {}
        for k, v in word2mor.items():
            mor_li = v
            if len(mor_li) > 3:
                mor_li = mor_li[:3]
            if len(mor_li) == 1:
                mor_li = mor_li + ['[mor-PAD-1]', '[mor-PAD-2]']
            if len(mor_li) == 2:
                mor_li = mor_li + ['[mor-PAD-2]']
            assert len(mor_li) == 3
            word2morid[k] = [mor2id.get(mor, mor2id['[mor-UNK]']) for mor in mor_li]

        for w in (set(word2id.keys()) - set(word2mor.keys())):
            word2morid[w] = [mor2id[w]] + [mor2id['[mor-PAD-1]'], mor2id['[mor-PAD-2]']]

        co_matrix = []
        for kv in sorted(word2id.items(), key=lambda kv: (kv[1], kv[0])):
            co_matrix.append(word2morid.get(kv[0]))
        co_matrix = torch.LongTensor(co_matrix)

        print("********************", lang, "dict")
        print("********************", "num_morphemes", len(mor2id), "num_words", len(word2id))
        # print("********************", set(word2id.keys()) - set(word2morid.keys()))

        return len(mor2id), len(word2id), co_matrix


    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
