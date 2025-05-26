import pdb
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def normalize(W):
    W_norm = torch.norm(W)
    W_norm = torch.relu(W_norm - 1) + 1
    W = W / W_norm
    return W


class SamePadConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, gamma=0.9, device=None):
        super().__init__()

        self.device = device
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups, bias=False
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]), requires_grad=True)
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.grad_dim, self.shape = [], []
        for p in self.conv.parameters():
            self.grad_dim.append(p.numel())
            self.shape.append(p.size())
        self.dim = sum(self.grad_dim)
        self.in_channels = in_channels
        self.out_features = out_channels

        self.n_chunks = in_channels
        self.chunk_in_d = self.dim // self.n_chunks
        self.chunk_out_d = int(in_channels * kernel_size // self.n_chunks)

        self.grads = torch.Tensor(sum(self.grad_dim)).fill_(0).to(self.device)
        self.f_grads = torch.Tensor(sum(self.grad_dim)).fill_(0).to(self.device)
        nh = 64
        self.controller = nn.Sequential(nn.Linear(self.chunk_in_d, nh), nn.SiLU())
        self.calib_w = nn.Linear(nh, self.chunk_out_d)
        self.calib_b = nn.Linear(nh, out_channels // in_channels)
        self.calib_f = nn.Linear(nh, out_channels // in_channels)
        dim = self.n_chunks * (self.chunk_out_d + 2 * out_channels // in_channels)
        self.W = nn.Parameter(torch.empty(dim, 32), requires_grad=False)
        nn.init.xavier_uniform_(self.W.data)
        self.W.data = normalize(self.W.data)

        # self.calib_w = torch.nn.Parameter(torch.ones(out_channels, in_channels,1), requires_grad = True)
        # self.calib_b = torch.nn.Parameter(torch.zeros([out_channels]), requires_grad = True)
        # self.calib_f = torch.nn.Parameter(torch.ones(1,out_channels,1), requires_grad = True)

        # 确定是否需要移除最后一个像素
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        self.gamma = gamma
        self.f_gamma = 0.3
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.trigger = 0
        self.tau = 0.75

    def ctrl_params(self):

        c_iter = chain(self.controller.parameters(), self.calib_w.parameters(),
                       self.calib_b.parameters(), self.calib_f.parameters())
        for p in c_iter:
            yield p

    def store_grad(self):

        # 实现梯度存储逻辑
        grad = self.conv.weight.grad.data.clone()
        grad = nn.functional.normalize(grad)
        grad = grad.view(-1)
        self.f_grads = self.f_gamma * self.f_grads + (1 - self.f_gamma) * grad
        self.grads = self.gamma * self.grads + (1 - self.gamma) * grad
        if not self.training:
            e = self.cos(self.f_grads, self.grads)
            if e < -self.tau:
                self.trigger = 1

    def fw_chunks(self):

        # 实现前向传播的块处理逻辑
        x = self.grads.view(self.n_chunks, -1)
        rep = self.controller(x)
        w = self.calib_w(rep)
        b = self.calib_b(rep)
        f = self.calib_f(rep)
        q = torch.cat([w.view(-1), b.view(-1), f.view(-1)])
        if not hasattr(self, 'q_ema'):
            setattr(self, 'q_ema', torch.zeros(*q.size()).float().to(self.device))

        self.q_ema = self.f_gamma * self.q_ema + (1 - self.f_gamma) * q
        q = self.q_ema
        if self.trigger == 1:
            dim = w.size(0)
            self.trigger = 0
            # read

            att = q @ self.W
            att = F.softmax(att / 0.5)

            v, idx = torch.topk(att, 2)
            ww = torch.index_select(self.W, 1, idx)
            # idx = idx.unsqueeze(1).float()
            old_w = ww @ v.clone().detach().unsqueeze(1).float()
            # old_w = ww @ idx
            # write memory
            s_att = torch.zeros(att.size(0)).to(self.device)
            s_att[idx.squeeze().long()] = v.squeeze()
            W = old_w @ s_att.unsqueeze(0)
            mask = torch.ones(W.size()).to(self.device)
            mask[:, idx.squeeze().long()] = self.tau
            self.W.data = mask * self.W.data + (1 - mask) * W
            self.W.data = normalize(self.W.data)
            # retrieve
            ll = torch.split(old_w, dim)
            nw, nb, nf = w.size(1), b.size(1), f.size(1)
            o_w, o_b, o_f = torch.cat(*[ll[:nw]]), torch.cat(*[ll[nw:nw + nb]]), torch.cat(*[ll[-nf:]])

            try:
                w = self.tau * w + (1 - self.tau) * o_w.view(w.size())
                b = self.tau * b + (1 - self.tau) * o_b.view(b.size())
                f = self.tau * f + (1 - self.tau) * o_f.view(f.size())
            except:
                pdb.set_trace()
        f = f.view(-1).unsqueeze(0).unsqueeze(2)

        return w.unsqueeze(0), b.view(-1), f

    def forward(self, x):

        w, b, f = self.fw_chunks()

        cw = self.conv.weight * w
        try:
            conv_out = F.conv1d(x, cw, padding=self.padding, dilation=self.dilation, bias=self.bias * b)
            out = f * conv_out
        except:
            pdb.set_trace()
        return out

    def representation(self, x):

        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

    def _forward(self, x):

        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False, gamma=0.9, device=None):
        super().__init__()
        self.device = device
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma,
                                 device=self.device)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma,
                                 device=self.device)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def ctrl_params(self):
        c_iter = chain(self.conv1.controller.parameters(), self.conv1.calib_w.parameters(),
                       self.conv1.calib_b.parameters(), self.conv1.calib_f.parameters(),
                       self.conv2.controller.parameters(), self.conv2.calib_w.parameters(),
                       self.conv2.calib_b.parameters(), self.conv2.calib_f.parameters())

        return c_iter

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, gamma=0.9, device=None):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1), gamma=gamma,
                device=self.device
            )
            for i in range(len(channels))
        ])

    def ctrl_params(self):
        ctrl = []
        for l in self.net:
            ctrl.append(l.ctrl_params())
        c = chain(*ctrl)
        for p in c:
            yield p

    def forward(self, x):
        return self.net(x)


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial', gamma=0.9, device=None):
        super().__init__()
        self.device = device
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3, gamma=gamma,
            device=self.device
        )
        self.repr_dropout = nn.Dropout(p=0.1)

        # [64] * 10 + [320] = [64, 64, 64, 64, 64, 64, 64, 64, 64 ,64, 320] = 11 items
        # for i in range(len(...)) -> 0, 1, ..., 10

    def ctrl_params(self):
        return self.feature_extractor.ctrl_params()

    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x

    def forward_time(self, x, mask=None):  # x: B x T x input_dims
        x = x.transpose(1, 2)
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x
