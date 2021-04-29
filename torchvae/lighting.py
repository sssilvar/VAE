import pytorch_lightning as pl
import torch.optim as optim

from .models import VAE


class LitVAE(pl.LightningModule):
    def __init__(self, in_features, hidden_features):
        super(LitVAE, self).__init__()
        self.save_hyperparameters()

        self.model = VAE(in_features, hidden_features)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def kld(self):
        return self.model.kld

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)
        nll = -y_pred.log_prob(x).sum()
        loss = self.kld + nll

        self.log('Train/Loss', loss, prog_bar=True)
        self.log('Train/KLD', self.kld, prog_bar=True)
        self.log('Train/NLL', nll, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)
        nll = -y_pred.log_prob(x).sum()
        loss = self.kld - nll

        self.log('Validation/Loss', loss, prog_bar=True)
        self.log('Validation/KLD', self.kld, prog_bar=True)
        self.log('Validation/NLL', nll, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters())


def train_vae(model, min_epochs, max_epochs, train_loader, val_loader=None, logger=True, logger_path='model_logs'):
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger

    n_gpus = torch.cuda.device_count()
    auto_select_gpus = True if n_gpus >= 2 else False

    loss_monitor_key = 'Validation/Loss' if val_loader is not None else 'Train/Loss'
    early_stop_loss = EarlyStopping(loss_monitor_key, patience=5, mode='min')

    if logger:
        logger = TensorBoardLogger(logger_path, name='VAE')
    trainer = Trainer(min_epochs=min_epochs, max_epochs=max_epochs, logger=logger,
                      callbacks=[early_stop_loss],
                      gpus=n_gpus, auto_select_gpus=auto_select_gpus)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    if logger is not None and logger is not False:
        logger.finalize('success')
    return model
