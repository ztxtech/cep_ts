import argparse
import importlib
import traceback
from copy import deepcopy

import torch
from torch import nn


def model_debug(func):
    def wrapper(self, *args, **kwargs):

        result = func(self, *args, **kwargs)

        if self.debug and self.args.stage == 'test':

            x = args[0]
            c = args[-1]

            for idx, f in enumerate(self.f_pool[c]):
                if not f.active:
                    continue
                # model traj update
                if c not in self.traj:
                    self.traj[c] = {}
                    self.f_hist[c] = {}

                if idx not in self.traj[c]:
                    self.traj[c][idx] = []
                    self.f_hist[c][idx] = []

                self.traj[c][idx].append([float(f.token[0]),
                                          float(f.token[1]),
                                          int(f.token[2]),
                                          self.router.iter[c] - 1])

                if f.last_time == self.router.iter[c] - 1:
                    self.f_hist[c][idx].append(self.router.iter[c] - 1)

            # sample_token update
            token = self.router.get_token(x)
            if c not in self.sample_token:
                self.sample_token[c] = []
            self.sample_token[c].append([float(token[0]),
                                         float(token[1]),
                                         self.router.iter[c] - 1])

        return result

    return wrapper


class Wrapper(nn.Module):
    def __init__(self, forecaster):
        super().__init__()
        self.forecaster = forecaster
        self.args = forecaster.args
        self.token = None
        self.slow_token = None
        self.fast_token = None
        self.last_time = None
        self.active = True

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        out = self.forecaster(batch_x, batch_x_mark, batch_y, batch_y_mark)
        return out

    def clone(self):
        try:
            wrapper = deepcopy(self)
            wrapper.args_sync(self.args)
            wrapper.token = None
            wrapper.slow_token = None
            wrapper.fast_token = None
            wrapper.last_time = None
        except:
            print(traceback.format_exc())
            print('deepcopy error')
            try:
                f = self.forecaster.__class__(self.args)
                f.load_state_dict(self.forecaster.state_dict())
                f.eval()
                wrapper = Wrapper(f)
            except:
                print('copy from state dict error')
                print(traceback.format_exc())
        return wrapper

    def die(self):
        self.active = False
        self.forecaster = None

    def args_sync(self, args):
        self.args = args
        self.forecaster.args = args


class Router(nn.Module):
    def __init__(self, args):
        super(Router, self).__init__()
        self.args = args
        self.mean_b = args.mean_b
        self.std_b = args.std_b
        self.iter = [0 for _ in range(args.enc_in)]
        self.wait_time = args.wait_time
        self.die_ratio = args.die_ratio
        self.replace_ratio = args.replace_ratio
        self.fast_ratio = args.fast_ratio
        self._set_scope()

    def _set_scope(self):
        if self.args.scope == 'seq':
            self.scope = self.args.seq_len

        if self.args.scope == 'pred':
            self.scope = self.args.pred_len

    def args_sync(self, args):
        self.args = args
        self.mean_b = args.mean_b
        self.std_b = args.std_b
        self.iter = [0 for _ in range(args.enc_in)]
        self.wait_time = args.wait_time
        self.die_ratio = args.die_ratio
        self.replace_ratio = args.replace_ratio
        self.fast_ratio = args.fast_ratio
        self._set_scope()

    def get_token(self, x):
        return [torch.mean(x, dim=(1, 2)), torch.std(x, dim=(1, 2)) + 1e-8]

    def distance(self, x_token, f):
        if f.active:
            x_mean, x_std = x_token
            f_mean, f_std, _ = f.token
            return ((x_mean - f_mean) ** 2 + (x_std - f_std) ** 2) ** 0.5
        else:
            return float('inf')

    def split(self, x_token, f_token):
        x_mean, x_std = x_token
        f_mean, f_std, f_n = f_token
        mean_split = torch.abs(x_mean - f_mean) > self.mean_b * f_std if self.mean_b != 0.0 else False
        std_split = (x_std < self.std_b * f_std) or (x_std > 1.0 / self.std_b * f_std) if self.std_b != 0.0 else False
        time_split = f_n > self.wait_time
        return (mean_split or std_split) and time_split

    def find_f(self, batch_x, pool, c):
        x_token = self.get_token(batch_x[:, -self.scope:, :])
        if pool[0].token is None or self.args.stage != 'test':
            f = pool[0]
        else:
            pool_d = [self.distance(x_token, f) for f in pool]
            _, index = torch.min(torch.tensor(pool_d), dim=0)
            if self.split(x_token, pool[index].token):
                f = pool[index].clone()
                pool.append(f)
                self.update_opt(f, pool[index])
            else:
                f = pool[index]

        self.update(f, x_token)

        if self.args.stage == 'test':
            f.last_time = self.iter[c]
            self.filter(pool, c)
            self.iter[c] += 1

            if self.args.pre_shift:
                y_token = self.get_token(self.args.batch_y[:, -self.scope:, :])
                pool_d = [self.distance(y_token, f) for f in pool]
                _, index = torch.min(torch.tensor(pool_d), dim=0)
                next_split = self.split(y_token, pool[index].token)

                for p in f.parameters():
                    p.requires_grad = not next_split
        return f

    def filter(self, pool, c):
        if self.die_ratio > 0.0:
            for f in pool:
                if f.last_time < self.iter[c] - max(2, self.die_ratio * f.token[2]):
                    f.die()

    def update(self, f, token):
        if self.args.stage == 'test':
            batch = len(token[0])
            for b in range(batch):
                item = [token[0][b], token[1][b]]
                if f.token is None:
                    f.slow_token = item + [1]
                    f.fast_token = item + [1]
                    f.token = item + [1]
                else:
                    n_old = f.token[2]
                    n_new = n_old + 1

                    slow_mean_old = f.slow_token[0]
                    slow_std_old = f.slow_token[1]
                    slow_mean_new = (n_old * slow_mean_old + item[0]) / (n_old + 1)
                    slow_std_new = torch.sqrt(
                        (n_old * slow_std_old ** 2 + item[1] ** 2 + (slow_mean_old - item[0]) ** 2) / (n_old + 1))
                    f.slow_token = [slow_mean_new, slow_std_new, n_new]

                    fast_mean_old = f.fast_token[0]
                    fast_std_old = f.fast_token[1]
                    fast_mean_new = (1 - self.replace_ratio) * fast_mean_old + self.replace_ratio * item[0]
                    fast_std_new = (1 - self.replace_ratio) * fast_std_old + self.replace_ratio * item[1]
                    f.fast_token = [fast_mean_new, fast_std_new, n_new]

                    mean_new = (1 - self.fast_ratio) * slow_mean_new + self.fast_ratio * fast_mean_new
                    std_new = (1 - self.fast_ratio) * slow_std_new + self.fast_ratio * fast_std_new

                    f.token = [mean_new, std_new, n_new]

    def update_opt(self, cur_f, near_f):
        opt = self.args.opt
        opt_mode = self.args.opt_mode
        new_param = cur_f.named_parameters()

        if not opt_mode:
            new_group = {'params': [],
                         'lr': self.args.learning_rate}
            for name, param in new_param:
                new_group['params'].append(param)
            opt.add_param_group(new_group)
            return

        if opt_mode == 'flat':
            opt_params = opt.param_groups[0]['params']
            for name, param in new_param:
                opt_params.append(param)
            return

        near_param = list(near_f.named_parameters())
        if opt_mode == 'fade':
            scale = self.args.opt_scale
            wait_time = self.args.opt_wait
            for param_group in opt.param_groups:
                if param_group['lr'] != self.args.learning_rate:
                    if wait_time > 0:
                        param_group['lr'] = max(param_group['lr'] * ((1.0 / scale) ** (1.0 / wait_time)),
                                                self.args.learning_rate)
                    else:
                        param_group['lr'] = self.args.learning_rate

            new_group = None
            for param_group in opt.param_groups:
                name, param = near_param[0]
                if any(p is param for p in param_group['params']):
                    new_group = deepcopy(param_group)
                    break

            new_group['params'] = []
            new_group['lr'] = self.args.learning_rate * scale
            for name, param in new_param:
                new_group['params'].append(param)
            if new_group is not None:
                opt.add_param_group(new_group)
            else:
                assert 'ERROR'
            return


class net(nn.Module):
    def __init__(self, args):
        super(net, self).__init__()
        self.args = args
        self.batch = self.args.batch_size
        self.seq_len = self.args.seq_len
        self.pred_len = self.args.pred_len
        self.enc_in = self.args.enc_in
        self.debug = self.args.debug

        self._net_init()

    def _get_forecaster(self):
        last_dot_index = self.args.forecaster.rfind('.')
        file = self.args.forecaster[:last_dot_index]
        module = self.args.forecaster[last_dot_index + 1:]
        f_class = getattr(importlib.import_module(f'{file}'), module)
        f_args = argparse.Namespace(**vars(self.args))
        f_args.enc_in = 1
        f = f_class(f_args)
        return f

    def _net_init(self):
        self.router = Router(self.args)
        self.f_pool = nn.ModuleList(nn.ModuleList() for _ in range(self.enc_in))

        for c in range(self.enc_in):
            f = self._get_forecaster()
            wrapper_f = Wrapper(f)
            self.f_pool[c].append(wrapper_f)

        if self.debug:
            self.traj = {}
            self.sample_token = {}
            self.f_hist = {}

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        batch = batch_x.size(0)
        outputs = torch.zeros((batch, self.pred_len, self.enc_in), dtype=batch_x.dtype).to(batch_x.device)

        for c in range(self.enc_in):
            x_c = batch_x[:, :, c].unsqueeze(2)
            y_c = batch_y[:, :, c].unsqueeze(2)
            outputs[:, :, c] = self._forward_channel(x_c, batch_x_mark, y_c, batch_y_mark, c).squeeze(2)
        return outputs

    @model_debug
    def _forward_channel(self, batch_x, batch_x_mark, batch_y, batch_y_mark, c):
        f = self.router.find_f(batch_x, self.f_pool[c], c)
        y = f(batch_x, batch_x_mark, batch_y, batch_y_mark)
        return y

    def args_sync(self, args):
        self.args = args
        self.router.args_sync(args)
        for pool in self.f_pool:
            for wrapper in pool:
                wrapper.args_sync(args)
