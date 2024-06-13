import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from .optim import warmup_lambda


class RegPL(pl.LightningModule):
    '''
    abstract class for regression task
    '''
    def __init__(self, model_config, optim_config) -> None:
        super(RegPL, self).__init__()
        self.model = None
        self.loss_fun = mse_loss
        self.optim_config = optim_config
        self.noise_var = optim_config.noise_var

    def forward(self, batch):
        return self.model(batch)
        
    def training_step(self, batch, batch_idx):
        rad, ws = batch
        ws_hat = self(rad)
        ws = ws[:, None]
        l = self.loss_fun(ws_hat, ws)
        self.log_dict(
            {
                "train/loss": l,
            }, 
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        return l

    def validation_step(self, batch, batch_idx):
        rad, ws = batch
        ws_hat = self(rad)
        ws = ws[:, None]
        l = mse_loss(ws_hat, ws)
        self.log_dict(
            {
                "val/loss": l,
            }, 
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )
    
    def test_step(self, batch, batch_idx):
        rad, ws = batch
        if not hasattr(self, "truth"):
            self.truth = []
            self.pred = []
        ws_hat = self(rad)
        ws = ws[:, None]
        l = mse_loss(ws_hat, ws)
        self.truth.append(ws.detach().cpu())
        self.pred.append(ws_hat.detach().cpu())
        self.log_dict(
            {
                "test/loss": l,
            }, 
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )

    def configure_optimizers(self):
        lr = self.optim_config.lr
        betas = self.optim_config.betas
        weight_decay = self.optim_config.weight_decay
        total_num_steps = self.optim_config.total_num_steps
        opt = torch.optim.Adam(
            params=self.model.parameters(),
            lr=lr, 
            betas=betas,
            weight_decay=weight_decay
        )

        warmup_iter = int(
            np.round(
                self.optim_config.warmup_percentage * total_num_steps)
        )
        if self.optim_config.lr_scheduler_mode == 'none':
            return opt
        else:
            if self.optim_config.lr_scheduler_mode == 'cosine':
                warmup_scheduler = LambdaLR(
                    opt,
                    lr_lambda=warmup_lambda(
                        warmup_steps=warmup_iter,
                        min_lr_ratio=self.optim_config.warmup_min_lr_ratio
                    )
                )
                cosine_scheduler = CosineAnnealingLR(
                    opt,
                    T_max=(total_num_steps - warmup_iter),
                    eta_min=self.optim_config.min_lr_ratio * self.optim_config.lr
                )
                lr_scheduler = SequentialLR(
                    opt,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_iter]
                )
                lr_scheduler_config = {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1, 
                }
            else:
                raise NotImplementedError
            return {
                    "optimizer": opt, 
                    "lr_scheduler": lr_scheduler_config
            }

   
class ClassiPL(pl.LightningModule):
    '''
    abstract class for classification task
    '''
    def __init__(self, model_config, optim_config) -> None:
        super(ClassiPL, self).__init__()
        self.model = None
        self.loss = CrossEntropyLoss()
        self.optim_config = optim_config
        self.noise_var = optim_config.noise_var

    def forward(self, batch):
        return self.model(batch)
        
    def classify(self, ws):
        category = torch.zeros_like(ws, dtype=int, device=ws.device)
        category[ws < 27] = 0
        category[(ws >= 27) & (ws < 45)] = 1
        category[ws >= 45] = 2
        return category
    
    def training_step(self, batch, batch_idx):
        rad, ws = batch
        pred = self(rad)
        truth = self.classify(ws)
        l = self.loss(pred, truth)
        self.log_dict(
            {
                "train/loss": l,
            }, 
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        return l

    def validation_step(self, batch, batch_idx):
        rad, ws = batch
        pred = self(rad)
        truth = self.classify(ws)
        l = self.loss(pred, truth)
        pred = torch.argmax(pred, dim=1)
        accurate = (pred == truth).sum().item() / pred.shape[0]
        self.log_dict(
            {
                "val/loss": l,
                "val/accu": accurate
            }, 
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )
    
    def test_step(self, batch, batch_idx):
        rad, ws = batch
        pred = self(rad)
        truth = self.classify(ws)
        l = self.loss(pred, truth)
        pred = torch.argmax(pred, dim=1)
        accurate = (pred == truth).sum().item() / pred.shape[0]
        self.log_dict(
            {
                "val/loss": l,
                "val/accu": accurate
            }, 
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True
        )

    def configure_optimizers(self):
        lr = self.optim_config.lr
        betas = self.optim_config.betas
        weight_decay = self.optim_config.weight_decay
        total_num_steps = self.optim_config.total_num_steps
        opt = torch.optim.Adam(
            params=self.model.parameters(),
            lr=lr, 
            betas=betas,
            weight_decay=weight_decay
        )

        warmup_iter = int(
            np.round(
                self.optim_config.warmup_percentage * total_num_steps)
        )
        if self.optim_config.lr_scheduler_mode == 'none':
            return opt
        else:
            if self.optim_config.lr_scheduler_mode == 'cosine':
                warmup_scheduler = LambdaLR(
                    opt,
                    lr_lambda=warmup_lambda(
                        warmup_steps=warmup_iter,
                        min_lr_ratio=self.optim_config.warmup_min_lr_ratio
                    )
                )
                cosine_scheduler = CosineAnnealingLR(
                    opt,
                    T_max=(total_num_steps - warmup_iter),
                    eta_min=self.optim_config.min_lr_ratio * self.optim_config.lr
                )
                lr_scheduler = SequentialLR(
                    opt,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_iter]
                )
                lr_scheduler_config = {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1, 
                }
            else:
                raise NotImplementedError
            return {
                    "optimizer": opt, 
                    "lr_scheduler": lr_scheduler_config
            }