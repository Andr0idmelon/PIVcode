import pytorch_lightning as pl
from torch import optim
import torchmetrics 
from model import FlowNet
import torch


class autoencoder(pl.LightningModule):

    def __init__(self,params):
        super().__init__()
        self.lr=params.training.lr
        self.model=FlowNet()
        self.save_hyperparameters(params)
        self.loss_fn=torchmetrics.MeanAbsoluteError()


    def forward(self,x):
        return self.model(x)


    def configure_optimizers(self):
        optimizer=optim.Adam(self.parameters(),lr=self.lr)
        milestones=[1000,2000]
        scheduler = {'scheduler': optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5),'interval': 'epoch' }
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict

    def training_step(self, batch, batch_idx):
        x, y = batch
        u_true = y[:, 0:1, :, :]
        v_true = y[:, 1:2, :, :] 
        u_pred = self.model(x)[:,0:1,:,:]
        v_pred = self.model(x)[:,1:2,:,:]

        loss_u = self.loss_fn(u_pred, u_true)
        loss_v = self.loss_fn(v_pred, v_true)
        loss_total = 0.5 * loss_u + 0.5 * loss_v

        self.log('train_loss', loss_total, sync_dist=True)

        return loss_total

    def validation_step(self, batch, batch_idx):
        x, y = batch
        u_true = y[:, 0:1, :, :]
        v_true = y[:, 1:2, :, :]
        u_pred = self.model(x)[:,0:1,:,:]
        v_pred = self.model(x)[:,1:2,:,:]
        
        loss_u = self.loss_fn(u_pred, u_true)
        loss_v = self.loss_fn(v_pred, v_true)
        loss_total = 0.5 * loss_u + 0.5 * loss_v

        self.log('val_loss', loss_total, sync_dist=True)

        return loss_total

