import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        # get parameters
        self.seq_len = args.seq_len
        self.enc_in = args.enc_in
        self.d_model = args.d_model
        self.dropout = args.dropout

        self.pred_len = args.pred_len

        self.seg_len = args.seg_len
        self.seg_num_x = max(self.seq_len // self.seg_len, 1)
        self.seg_num_y = max(self.pred_len // self.seg_len, 1)

        # building model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = batch_x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        seq_last = batch_x[:, -1:, :].detach()
        batch_x = (batch_x - seq_last).permute(0, 2, 1)  # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        batch_x = self.valueEmbedding(batch_x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        _, hn = self.rnn(batch_x)  # bc,n,d  1,bc,d

        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))  # bcm,1,d  1,bcm,d

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_last
        return y[:, -self.pred_len:, :]
