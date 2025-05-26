import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from joblib.externals.loky.backend import get_context
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from utils.metrics import metric, cumavg
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'partial']
        self.n_inner = args.n_inner

    def _get_data(self, flag):
        args = self.args

        data_dict_ = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = args.test_bsz
            freq = args.freq
        elif flag == 'val':
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
            freq = args.detail_freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            delay_fb=args.delay_fb,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            prefetch_factor=args.prefetch_factor if args.num_workers > 1 else None,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else None,
            pin_memory_device=f'cuda:{self.args.gpu}' if args.num_workers > 0 and args.use_gpu else '',
            multiprocessing_context=get_context('loky') if args.num_workers > 0 else None
        )
        return data_set, data_loader

    def _select_optimizer(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        self.args.device = device
        return device

    def _get_model(self):
        pass

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = Path(self.args.out_path) / self.args.checkpoints / setting
        os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.opt = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.opt.zero_grad()

                pred, true = self._train_forward(
                    batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    self.opt.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, ))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.opt, epoch + 1, self.args)

        best_model_path = path / 'checkpoint.pth'
        self.model = torch.load(best_model_path, weights_only=False)

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._train_forward(
                batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        path = Path(self.args.out_path) / self.args.checkpoints / setting
        self.opt = self._select_optimizer()
        test_data, test_loader = self._get_data(flag='test')

        if self.online == 'none':
            for p in self.model.parameters():
                p.requires_grad = False

        preds = []
        trues = []
        start = time.time()
        maes, mses = [], []

        bar = tqdm(test_loader)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(bar):
            pred, true = self._ol_forward(
                batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            mae, mse = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())

            maes.append(mae)
            mses.append(mse)
            bar.set_postfix({"mse": mse, "mae": mae})

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('test shape:', preds.shape, trues.shape)
        MAE, MSE = cumavg(maes), cumavg(mses)
        mae, mse = MAE[-1], MSE[-1]

        end = time.time()
        exp_time = end - start

        print('mse:{}, mae:{}, time:{}'.format(mse, mae, exp_time))

        del self.args.device

        torch.save(self.model, path / 'final.pth')

        return [mae, mse, exp_time], MAE, MSE, preds, trues

    def _train_forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        trues = batch_y[:, -self.args.pred_len:, :].float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        return outputs, trues

    def _ol_forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        criterion = self._select_criterion()

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        trues = batch_y[:, -self.args.pred_len:, :].float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        for i in range(self.n_inner):
            self.opt.zero_grad()

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if i == 0:
                output_ol = outputs.clone().detach().cpu()

            loss = criterion(outputs, trues)
            if outputs.requires_grad:
                loss.backward()
                self.opt.step()

        return output_ol, trues
