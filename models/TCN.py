import torch
import torch.nn as nn

from layers.FSNetBlock import TSEncoder


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)[:, -1]


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.pred_len = args.pred_len
        encoder = TSEncoder(input_dims=args.enc_in + 4,
                            output_dims=320,  # standard ts2vec backbone value
                            hidden_dims=64,  # standard ts2vec backbone value
                            depth=10,
                            device=self.device)
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.regressor = nn.Linear(320, self.enc_in * self.pred_len).to(self.device)

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        x = torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1)
        rep = self.encoder(x)
        y = self.regressor(rep)
        out = y.reshape(-1, self.pred_len, self.enc_in)
        return out
