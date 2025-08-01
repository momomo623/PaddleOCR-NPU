
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchocr.modeling.common import Activation
import numpy as np
from .self_attention import WrapEncoderForFeature
from .self_attention import WrapEncoder

from collections import OrderedDict
gradient_clip = 10

# https://forums.fast.ai/t/lambda-layer/28507/5
class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x):
        return self.func(x)

class PVAM(nn.Module):
    def __init__(self, in_channels, char_num, max_text_length, num_heads,
                 num_encoder_tus, hidden_dims):
        super(PVAM, self).__init__()
        self.char_num = char_num
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_tus
        self.hidden_dims = hidden_dims
        # Transformer encoder
        t = 256
        c = 512
        self.wrap_encoder_for_feature = WrapEncoderForFeature(
            src_vocab_size=1,
            max_length=t,
            n_layer=self.num_encoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.0,#0.1,
            attention_dropout=0.0,#0.1,
            relu_dropout=0.0,#0.1,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)

        # PVAM
        self.flatten0 = Lambda(lambda x: torch.flatten(x, start_dim=0, end_dim=1))
        self.fc0 = torch.nn.Linear(
            in_features=in_channels,
            out_features=in_channels, )
        self.emb = torch.nn.Embedding(
            num_embeddings=self.max_length, embedding_dim=in_channels)
        self.flatten1 = Lambda(lambda x: torch.flatten(x, start_dim=0, end_dim=2))
        self.fc1 = torch.nn.Linear(
            in_features=in_channels, out_features=1, bias=False)

    def forward(self, inputs, encoder_word_pos, gsrm_word_pos):
        b, c, h, w = inputs.shape
        conv_features = torch.reshape(inputs, shape=[-1, c, h * w])
        conv_features = conv_features.permute(0, 2, 1)
        # transformer encoder
        b, t, c = conv_features.shape

        enc_inputs = [conv_features, encoder_word_pos, None]
        word_features = self.wrap_encoder_for_feature(enc_inputs)

        # pvam
        b, t, c = word_features.shape
        word_features = self.fc0(word_features)
        word_features_ = torch.reshape(word_features, [-1, 1, t, c])
        word_features_ = word_features_.repeat([1, self.max_length, 1, 1])
        word_pos_feature = self.emb(gsrm_word_pos)
        word_pos_feature_ = torch.reshape(word_pos_feature,
                                           [-1, self.max_length, 1, c])
        word_pos_feature_ = word_pos_feature_.repeat([1, 1, t, 1])
        y = word_pos_feature_ + word_features_
        y = torch.tanh(y)
        attention_weight = self.fc1(y)
        attention_weight = torch.reshape(
            attention_weight, shape=[-1, self.max_length, t])
        attention_weight = F.softmax(attention_weight, dim=-1)
        pvam_features = torch.matmul(attention_weight,
                                      word_features)  #[b, max_length, c]
        return pvam_features


class GSRM(nn.Module):
    def __init__(self, in_channels, char_num, max_text_length, num_heads,
                 num_encoder_tus, num_decoder_tus, hidden_dims):
        super(GSRM, self).__init__()
        self.char_num = char_num
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_tus
        self.num_decoder_TUs = num_decoder_tus
        self.hidden_dims = hidden_dims

        self.fc0 = torch.nn.Linear(
            in_features=in_channels, out_features=self.char_num)
        self.wrap_encoder0 = WrapEncoder(
            src_vocab_size=self.char_num + 1,
            max_length=self.max_length,
            n_layer=self.num_decoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)

        self.wrap_encoder1 = WrapEncoder(
            src_vocab_size=self.char_num + 1,
            max_length=self.max_length,
            n_layer=self.num_decoder_TUs,
            n_head=self.num_heads,
            d_key=int(self.hidden_dims / self.num_heads),
            d_value=int(self.hidden_dims / self.num_heads),
            d_model=self.hidden_dims,
            d_inner_hid=self.hidden_dims,
            prepostprocess_dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            preprocess_cmd="n",
            postprocess_cmd="da",
            weight_sharing=True)

        self.mul = lambda x: torch.matmul(x,
                                          self.wrap_encoder0.prepare_decoder.emb0.weight.t(),
                                          )

    def forward(self, inputs, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2):
        # ===== GSRM Visual-to-semantic embedding block =====
        b, t, c = inputs.shape
        pvam_features = torch.reshape(inputs, [-1, c])
        word_out = self.fc0(pvam_features)
        word_ids = torch.argmax(F.softmax(word_out, dim=-1), dim=1)
        word_ids = torch.reshape(word_ids, shape=[-1, t, 1])

        #===== GSRM Semantic reasoning block =====
        """
        This module is achieved through bi-transformers,
        ngram_feature1 is the froward one, ngram_fetaure2 is the backward one
        """
        pad_idx = self.char_num
        word1 = F.pad(word_ids.type(torch.float32), [0, 0, 1, 0, 0, 0], value=1.0 * pad_idx)
        word1 = word1.type(torch.int64)
        word1 = word1[:, :-1, :]
        word2 = word_ids

        enc_inputs_1 = [word1, gsrm_word_pos, gsrm_slf_attn_bias1]
        enc_inputs_2 = [word2, gsrm_word_pos, gsrm_slf_attn_bias2]

        gsrm_feature1 = self.wrap_encoder0(enc_inputs_1)
        gsrm_feature2 = self.wrap_encoder1(enc_inputs_2)

        gsrm_feature2 = F.pad(gsrm_feature2, [0, 0, 0, 1, 0, 0],
                              value=0.,
                              )
        gsrm_feature2 = gsrm_feature2[:, 1:, ]
        gsrm_features = gsrm_feature1 + gsrm_feature2

        gsrm_out = self.mul(gsrm_features)

        b, t, c = gsrm_out.shape
        gsrm_out = torch.reshape(gsrm_out, [-1, c])

        return gsrm_features, word_out, gsrm_out


class VSFD(nn.Module):
    def __init__(self, in_channels=512, pvam_ch=512, char_num=38):
        super(VSFD, self).__init__()
        self.char_num = char_num
        self.fc0 = torch.nn.Linear(
            in_features=in_channels * 2, out_features=pvam_ch)
        self.fc1 = torch.nn.Linear(
            in_features=pvam_ch, out_features=self.char_num)

    def forward(self, pvam_feature, gsrm_feature):
        b, t, c1 = pvam_feature.shape
        b, t, c2 = gsrm_feature.shape
        combine_feature_ = torch.cat([pvam_feature, gsrm_feature], dim=2)
        img_comb_feature_ = torch.reshape(
            combine_feature_, shape=[-1, c1 + c2])
        img_comb_feature_map = self.fc0(img_comb_feature_)
        img_comb_feature_map = torch.sigmoid(img_comb_feature_map)
        img_comb_feature_map = torch.reshape(
            img_comb_feature_map, shape=[-1, t, c1])
        combine_feature = img_comb_feature_map * pvam_feature + (
            1.0 - img_comb_feature_map) * gsrm_feature
        img_comb_feature = torch.reshape(combine_feature, shape=[-1, c1])

        out = self.fc1(img_comb_feature)
        return out


class SRNHead(nn.Module):
    def __init__(self, in_channels, out_channels, max_text_length, num_heads,
                 num_encoder_TUs, num_decoder_TUs, hidden_dims, **kwargs):
        super(SRNHead, self).__init__()
        self.char_num = out_channels
        self.max_length = max_text_length
        self.num_heads = num_heads
        self.num_encoder_TUs = num_encoder_TUs
        self.num_decoder_TUs = num_decoder_TUs
        self.hidden_dims = hidden_dims

        self.pvam = PVAM(
            in_channels=in_channels,
            char_num=self.char_num,
            max_text_length=self.max_length,
            num_heads=self.num_heads,
            num_encoder_tus=self.num_encoder_TUs,
            hidden_dims=self.hidden_dims)

        self.gsrm = GSRM(
            in_channels=in_channels,
            char_num=self.char_num,
            max_text_length=self.max_length,
            num_heads=self.num_heads,
            num_encoder_tus=self.num_encoder_TUs,
            num_decoder_tus=self.num_decoder_TUs,
            hidden_dims=self.hidden_dims)
        self.vsfd = VSFD(in_channels=in_channels, char_num=self.char_num)

        self.gsrm.wrap_encoder1.prepare_decoder.emb0 = self.gsrm.wrap_encoder0.prepare_decoder.emb0

    def forward(self, inputs, others):
        encoder_word_pos = others[0]
        gsrm_word_pos = others[1].type(torch.long)
        gsrm_slf_attn_bias1 = others[2]
        gsrm_slf_attn_bias2 = others[3]

        pvam_feature = self.pvam(inputs, encoder_word_pos, gsrm_word_pos)

        gsrm_feature, word_out, gsrm_out = self.gsrm(
            pvam_feature, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2)

        final_out = self.vsfd(pvam_feature, gsrm_feature)
        if not self.training:
            final_out = F.softmax(final_out, dim=1)

        _, decoded_out = torch.topk(final_out, k=1)

        predicts = OrderedDict([
            ('predict', final_out),
            ('pvam_feature', pvam_feature),
            ('decoded_out', decoded_out),
            ('word_out', word_out),
            ('gsrm_out', gsrm_out),
        ])

        return predicts
