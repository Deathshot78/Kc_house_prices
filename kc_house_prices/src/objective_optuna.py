import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger 
import optuna
from model import KcHousePrices 
from data_utils import KcHousePricesDataModule, KcHousepricesDataset

# Paths
preprocessed_csv_path = 'projects/kc_house_prices/data/kc_house_data_preprocessed.csv'

def objective(trial):
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    # --- Hyperparameter Suggestions ---
    num_layers = trial.suggest_int('num_layers', 1, 5)
    hidden_units = trial.suggest_int("hidden_units", 64, 512, step=64)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    lr_scheduler_patience = trial.suggest_int('lr_scheduler_patience', 2, 5)
    lr_scheduler_factor = trial.suggest_categorical('lr_scheduler_factor', [0.1, 0.2, 0.5])

    # --- Instantiate and Setup DataModule to get input_shape ---
    data_module = KcHousePricesDataModule(
        preprocessed_csv_path=preprocessed_csv_path,
        batch_size=64, 
        num_workers=0 
    )
    data_module.prepare_data() # Call prepare_data (e.g., to check file existence)
    data_module.setup(stage='fit')  # Manually call setup for the 'fit' stage

    if data_module.input_shape is None:
        # This should not happen if setup ran correctly and data was loaded
        print("Error: data_module.input_shape is None after setup. Check data loading.")
        return float('inf') # Indicate a failed trial

    # --- Create Model ---
    model = KcHousePrices(
        input_shape=data_module.input_shape, # Use input_shape from the DataModule
        hidden_units=hidden_units,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor
    )

    # --- Callbacks ---
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=False) 
    callbacks_list = [early_stop_callback]


    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=25,  # Adjusted slightly, ensure it's enough for convergence with early stopping
        logger=TensorBoardLogger("lightning_logs_optuna", name=f"trial_{trial.number}"),
        callbacks=callbacks_list,
        accelerator="cpu", # or "gpu"
        devices="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
    )

        # Train model
    trainer.fit(model, datamodule=data_module)

    # Return validation loss for Optuna to minimize
    return trainer.callback_metrics["val_loss"].item()

def create_optuna_study(n_trials : int):
    study = optuna.create_study(direction="minimize")
    # You can also control Optuna's own verbosity:
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Shows only Optuna warnings and errors
    optuna.logging.set_verbosity(optuna.logging.INFO) # Default, shows completion messages
    study.optimize(objective, n_trials=n_trials) # Or your desired number of trials
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("Best hyperparameters:", study.best_params)
    print("Best value (min val_loss):", study.best_value)

if __name__ == "__main__":
    create_optuna_study(n_trials=100)