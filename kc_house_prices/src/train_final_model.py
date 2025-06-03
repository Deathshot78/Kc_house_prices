import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from datetime import datetime
from model import KcHousePrices 
from data_utils import KcHousePricesDataModule 

# --- Configuration Constants ---
BEST_HPARAMS = {
    'num_layers': 2,
    'hidden_units': 192,
    'dropout': 0.13,
    'lr': 2e-4,
    'weight_decay': 1e-4,
    'lr_scheduler_patience': 4,
    'lr_scheduler_factor': 0.5,
    'batch_size': 32
}
MAX_EPOCHS_FINAL = 200
SEED = 42
PREPROCESSED_CSV_PATH = 'data/kc_house_data_preprocessed.csv'
ROOT_LOG_DIR = "logs/lightning_logs" # Root for all experiments

def train_model(hparams: dict,
                preprocessed_csv_path: str,
                base_run_dir: str, # e.g., "lightning_logs/kc_house_mlp_final_experiment"
                max_epochs: int,
                seed: int):
    """
    Trains the KcHousePrices model. Logs training/validation metrics to CSV.
    Saves the best model checkpoint.
    """
    print(f"\n--- Starting Model Training. Logs & Checkpoints in: {base_run_dir} ---")
    pl.seed_everything(seed, workers=True)

    checkpoint_save_path = os.path.join(base_run_dir, "model_checkpoints")
    os.makedirs(checkpoint_save_path, exist_ok=True)

    data_module = KcHousePricesDataModule(
        preprocessed_csv_path=preprocessed_csv_path,
        batch_size=hparams['batch_size'],
        num_workers=4,
        random_state=seed
    )
    data_module.prepare_data()
    data_module.setup(stage='fit')

    if data_module.input_shape is None:
        raise ValueError("Input shape not determined by DataModule.")

    model = KcHousePrices(
        input_shape=data_module.input_shape,
        hidden_units=hparams['hidden_units'],
        num_layers=hparams['num_layers'],
        dropout=hparams['dropout'],
        lr=hparams['lr'],
        weight_decay=hparams['weight_decay'],
        lr_scheduler_patience=hparams['lr_scheduler_patience'],
        lr_scheduler_factor=hparams['lr_scheduler_factor']
    )

    # CSVLogger for training logs (train_logs/version_X/metrics.csv)
    train_csv_logger = CSVLogger(
        save_dir=base_run_dir, 
        name="train_logs",     
        version=f"seed{seed}" 
    )
    print(f"Training CSV logs will be saved in: {train_csv_logger.log_dir}")

    early_stop_callback = EarlyStopping(monitor='val_mape_dollars', patience=20, mode='min', verbose=True)
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_save_path,
        filename='best-model-{epoch:02d}-{val_mape_dollars:.2f}',
        save_top_k=1,
        monitor='val_mape_dollars',
        mode='min',
        verbose=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    callbacks = [early_stop_callback, model_checkpoint_callback, lr_monitor_callback]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices='auto',
        logger=train_csv_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=True
    )

    trainer.fit(model, datamodule=data_module)
    print("--- Model Training Finished ---")

    best_model_path = model_checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        print("Warning: ModelCheckpoint did not save a best model path.")
        return None
    return best_model_path


def test_model(checkpoint_path: str,
               preprocessed_csv_path: str,
               batch_size_for_test: int,
               seed: int):
    """
    Loads a trained model from a checkpoint and tests it.
    Does NOT save test metrics to a CSV via a logger, prints to console.
    """
    print(f"\n--- Testing Model from Checkpoint: {checkpoint_path} ---")
    pl.seed_everything(seed, workers=True)

    try:
        model = KcHousePrices.load_from_checkpoint(checkpoint_path)
        print(f"Model loaded successfully from: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model from checkpoint '{checkpoint_path}': {e}")
        return None

    data_module = KcHousePricesDataModule(
        preprocessed_csv_path=preprocessed_csv_path,
        batch_size=batch_size_for_test,
        num_workers=4,
        random_state=seed
    )

    # Trainer for Testing 
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        logger=False, 
        enable_progress_bar=True,
        enable_model_summary=False
    )

    # The model's test_step will still use self.log(), but these metrics won't go to a CSV
    # unless a logger is attached to the trainer. They will be in the returned results.
    test_results = trainer.test(model=model, datamodule=data_module)
    
    print("--- Model Testing Finished (Function) ---")
    return test_results


if __name__ == "__main__":

    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_specific_name = f"kc_house_mlp_final_run_{current_time_str}" # More unique run name
    
    MAIN_RUN_DIR = os.path.join(ROOT_LOG_DIR, run_specific_name)
    os.makedirs(MAIN_RUN_DIR, exist_ok=True)
    print(f"All logs and checkpoints for this run will be under: {MAIN_RUN_DIR}")

    # 1. Train the model
    best_model_checkpoint_path = train_model(
        hparams=BEST_HPARAMS,
        preprocessed_csv_path=PREPROCESSED_CSV_PATH,
        base_run_dir=MAIN_RUN_DIR,
        max_epochs=MAX_EPOCHS_FINAL,
        seed=SEED
    )

    # 2. Test the best model
    if best_model_checkpoint_path:
        final_test_results = test_model(
            checkpoint_path=best_model_checkpoint_path,
            preprocessed_csv_path=PREPROCESSED_CSV_PATH,
            batch_size_for_test=BEST_HPARAMS['batch_size'], # Use batch_size from hparams
            seed=SEED
        )

    else:
        print("Training did not result in a valid model checkpoint. Skipping testing.")