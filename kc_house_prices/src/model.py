import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

class KcHousePrices(pl.LightningModule):

    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        num_layers: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        lr_scheduler_patience: int = 3,
        lr_scheduler_factor: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        layers.append(nn.Linear(input_shape, hidden_units))
        layers.append(nn.GELU())

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_units, 1))

        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.HuberLoss()

    def forward(self, x):
        return self.model(x).squeeze(1) # Output shape: [batch_size]

    def _common_step(self, batch, batch_idx, stage: str):
        x, log_y_true_batch = batch 
        
        log_y_hat = self(x) # Model output [batch_size]

        # Ensure log_y_true has same shape as log_y_hat for loss
        if log_y_true_batch.ndim == 2 and log_y_true_batch.shape[1] == 1:
            log_y_true = log_y_true_batch.squeeze(1)
        elif log_y_true_batch.ndim == 1:
            log_y_true = log_y_true_batch
        else:
            # Fallback or error for unexpected shape
            log_y_true = log_y_true_batch.squeeze() 


        loss = self.loss_fn(log_y_hat, log_y_true)

        safe_log_y = torch.where(
            torch.abs(log_y_true) < 1e-8,
            torch.ones_like(log_y_true) * 1e-8,
            log_y_true,
        )
        mape_log = torch.mean(torch.abs((log_y_hat - log_y_true) / safe_log_y)) * 100
        mae_log = torch.mean(torch.abs(log_y_hat - log_y_true))

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, sync_dist=True)
        # Only show log-space MAPE/MAE on progress bar if it's training, otherwise they are less interpretable
        self.log(f"{stage}_mape_log", mape_log, on_step=False, on_epoch=True, prog_bar=(stage == "train"), sync_dist=True)
        self.log(f"{stage}_mae_log", mae_log, on_step=False, on_epoch=True, prog_bar=(stage == "train"), sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        x, log_y_true_batch = batch

        log_y_hat = self(x) # Model output [batch_size]

        if log_y_true_batch.ndim == 2 and log_y_true_batch.shape[1] == 1:
            log_y_true = log_y_true_batch.squeeze(1)
        elif log_y_true_batch.ndim == 1:
            log_y_true = log_y_true_batch
        else:
            log_y_true = log_y_true_batch.squeeze()

        # 1) Compute and log log‐space loss & metrics
        loss = self.loss_fn(log_y_hat, log_y_true)
        safe_log_y = torch.where(
            torch.abs(log_y_true) < 1e-8,
            torch.ones_like(log_y_true) * 1e-8,
            log_y_true,
        )
        mape_log = torch.mean(torch.abs((log_y_hat - log_y_true) / safe_log_y)) * 100
        mae_log = torch.mean(torch.abs(log_y_hat - log_y_true))

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_mape_log", mape_log, prog_bar=False, on_epoch=True, sync_dist=True) # Not on prog bar
        self.log("val_mae_log", mae_log, prog_bar=False, on_epoch=True, sync_dist=True)  # Not on prog bar

        # 2) Exponentiate to get original price scale 
        price_pred = torch.expm1(log_y_hat)
        price_true = torch.expm1(log_y_true)

        mae_dollars = torch.mean(torch.abs(price_pred - price_true))
        safe_price_true = torch.where(
            price_true < 1e-8, 
            torch.ones_like(price_true) * 1e-8,
            price_true,
        )
        mape_dollars = torch.mean(torch.abs((price_pred - price_true) / safe_price_true)) * 100

        # Log USD‐space metrics 
        self.log("val_mae_dollars", mae_dollars, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_mape_dollars", mape_dollars, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, log_y_true_batch = batch

        log_y_hat = self(x) # Model output [batch_size]
        
        if log_y_true_batch.ndim == 2 and log_y_true_batch.shape[1] == 1:
            log_y_true = log_y_true_batch.squeeze(1)
        elif log_y_true_batch.ndim == 1:
            log_y_true = log_y_true_batch
        else:
            log_y_true = log_y_true_batch.squeeze()

        loss = self.loss_fn(log_y_hat, log_y_true)
        safe_log_y = torch.where(
            torch.abs(log_y_true) < 1e-8,
            torch.ones_like(log_y_true) * 1e-8,
            log_y_true,
        )
        mape_log = torch.mean(torch.abs((log_y_hat - log_y_true) / safe_log_y)) * 100
        mae_log = torch.mean(torch.abs(log_y_hat - log_y_true))

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_mape_log", mape_log, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("test_mae_log", mae_log, prog_bar=False, on_epoch=True, sync_dist=True)

        price_pred = torch.expm1(log_y_hat)
        price_true = torch.expm1(log_y_true)

        mae_dollars = torch.mean(torch.abs(price_pred - price_true))
        safe_price_true = torch.where(
            price_true < 1e-8,
            torch.ones_like(price_true) * 1e-8,
            price_true,
        )
        mape_dollars = torch.mean(torch.abs((price_pred - price_true) / safe_price_true)) * 100

        # Log USD‐space metrics 
        self.log("test_mae_dollars", mae_dollars, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_mape_dollars", mape_dollars, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # This monitors the log-space val_loss
                "interval": "epoch",
                "frequency": 1,
            },
        }