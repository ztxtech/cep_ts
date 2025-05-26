import importlib
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from exp.exp_base import Exp_Basic
from utils.metrics import metric, cumavg
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp_TS2VecSupervised(Exp_Basic):

    def __init__(self, args):
        super().__init__(args)
        self.model = self._get_model()

    def _get_model(self):
        last_dot_index = self.args.model.rfind('.')
        file = self.args.model[:last_dot_index]
        module = self.args.model[last_dot_index + 1:]
        model_class = getattr(importlib.import_module(f'{file}'), module)
        model = model_class(self.args).to(self.device)
        return model

    def train(self, setting):

        cache_path = Path(self.args.out_path) / 'DET_cache'
        cache_path.mkdir(exist_ok=True)
        model_cache = cache_path / f'{self.args.data}_{self.args.pred_len}_{self.args.forecaster}_{self.args.seed}.pth'

        if model_cache.exists():
            self.model = torch.load(model_cache, weights_only=False)
        else:
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
                self.args.stage = 'train'
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

                vali_loss = self.vali(vali_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, ))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                adjust_learning_rate(self.opt, epoch + 1, self.args)

            best_model_path = path / 'checkpoint.pth'
            self.model = torch.load(best_model_path, weights_only=False)
            shutil.copy(best_model_path, model_cache)

        self.model.args_sync(self.args)

        return self.model

    def vali(self, vali_loader, criterion):
        self.args.stage = 'vali'
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
        self.model.eval()
        path = Path(self.args.out_path) / self.args.checkpoints / setting
        os.makedirs(path, exist_ok=True)
        self.opt = self._select_optimizer()
        self.args.opt = self.opt

        self.args.stage = 'test'
        test_data, test_loader = self._get_data(flag='test')

        if self.online == 'partial':
            pass
        elif self.online == 'none':
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
        del self.args.stage
        del self.args.device
        del self.args.batch_y
        del self.args.opt

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
        self.args.batch_y = batch_y

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
