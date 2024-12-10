
from runcontrol import autoencoder
from pytorch_lightning.loggers import WandbLogger
from dataworkimport MyDataModule
from callbacks import set_callbacks
from pytorch_lightning import Trainer



def training(params):
    wandb_logger=WandbLogger(name='demo',project="PIV",log_model=True)
    data=MyDataModule(params)
    model=autoencoder(params)
    callbacks=set_callbacks(params)
    trainer=Trainer(max_epochs=params.training.training_epoch,callbacks=callbacks,logger=wandb_logger,precision=16,accelerator='gpu',devices=[0],strategy="dp",sync_batchnorm=True)
    trainer.fit(model,data)




    

