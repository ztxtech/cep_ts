from math import ceil

import torch
import torch.nn as nn
from einops import rearrange, repeat

from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.enc_in = args.enc_in
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.seg_len = 12
        self.win_size = 2

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * args.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * args.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (args.e_layers - 1)))
        self.head_nf = args.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(args.d_model, self.seg_len, self.seg_len,
                                                  self.pad_in_len - args.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, args.enc_in, self.in_seg_num, args.d_model))
        self.pre_norm = nn.LayerNorm(args.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(args, 1 if l is 0 else self.win_size, args.d_model, args.n_heads, args.d_ff,
                            1, args.dropout,
                            self.in_seg_num if l is 0 else ceil(self.in_seg_num / self.win_size ** l), args.factor
                            ) for l in range(args.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, args.enc_in, (self.pad_out_len // self.seg_len), args.d_model))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(args, (self.pad_out_len // self.seg_len), args.factor, args.d_model,
                                           args.n_heads,
                                           args.d_ff, args.dropout),
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=False),
                        args.d_model, args.n_heads),
                    self.seg_len,
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    # activation=args.activation,
                )
                for l in range(args.e_layers + 1)
            ],
        )

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        # embedding
        batch_x, n_vars = self.enc_value_embedding(batch_x.permute(0, 2, 1))
        batch_x = rearrange(batch_x, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        batch_x += self.enc_pos_embedding
        batch_x = self.pre_norm(batch_x)
        enc_out, attns = self.encoder(batch_x)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_x.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out[:, -self.pred_len:, :]
