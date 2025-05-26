import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(args.seq_len, args.d_model, args.embed, args.freq,
                                                    args.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=False), args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        # Decoder
        self.projection = nn.Linear(args.d_model, args.pred_len, bias=True)

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        # Normalization from Non-stationary Transformer
        means = batch_x.mean(1, keepdim=True).detach()
        batch_x = batch_x - means
        stdev = torch.sqrt(torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        batch_x /= stdev

        _, _, N = batch_x.shape

        # Embedding
        enc_out = self.enc_embedding(batch_x, batch_x_mark)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out[:, -self.pred_len:, :]
