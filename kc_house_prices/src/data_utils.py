import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from typing import Optional
import os

def load_and_prepare_data_splits(
    preprocessed_csv_path: str, 
    target_column_name: str = 'price_log', 
    test_size_initial: float = 0.3, 
    val_size_from_test: float = 0.5, 
    random_state: int = 42
):
    """
    Loads preprocessed data, separates features/target, converts to tensors, 
    and performs train-validation-test splits.
    """
    try:
        df_processed = pd.read_csv(preprocessed_csv_path)
        print(f"Successfully loaded preprocessed data from: {preprocessed_csv_path}")
    except FileNotFoundError:
        print(f"Error: Preprocessed data file not found at {preprocessed_csv_path}")
        return None, None, None, None, None, None, None # Added one more None for X_shape

    if target_column_name not in df_processed.columns:
        print(f"Error: Target column '{target_column_name}' not found.")
        return None, None, None, None, None, None, None

    y_data = df_processed[target_column_name]
    # Ensure 'price' (original scale) is also dropped if it exists and isn't the target
    columns_to_drop = ['price', target_column_name] if 'price' in df_processed.columns else [target_column_name]
    X_data = df_processed.drop(columns=columns_to_drop, errors='ignore')
    
    X_tensor = torch.tensor(X_data.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_data.values, dtype=torch.float32).unsqueeze(1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_tensor, y_tensor, test_size=test_size_initial, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size_from_test, random_state=random_state
    )
    
    print(f"Data loaded & split: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, X_data.shape[1]

class KcHousepricesDataset(Dataset): 
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y) 

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
    
class KcHousePricesDataModule(pl.LightningDataModule):
    def __init__(self, 
                 preprocessed_csv_path: str,
                 target_column_name: str = 'price_log',
                 batch_size: int = 32, 
                 num_workers: int = 0, 
                 random_state: int = 42
                ):
        super().__init__()
        self.preprocessed_csv_path = preprocessed_csv_path
        self.target_column_name = target_column_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        
        # Placeholders for datasets and input_shape
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self.input_shape: Optional[int] = None 
        self.save_hyperparameters() # Saves __init__ args

    def prepare_data(self):
        if not os.path.exists(self.preprocessed_csv_path):
            raise FileNotFoundError(f"Preprocessed data not found at {self.preprocessed_csv_path}. Please run preprocess.py first.")

    def setup(self, stage: Optional[str] = None):
        X_train, y_train, X_val, y_val, X_test, y_test, input_shape_val = load_and_prepare_data_splits(
            self.preprocessed_csv_path,
            target_column_name=self.target_column_name,
            random_state=self.random_state
        )
        self.input_shape = input_shape_val # Store for model initialization
        
        if stage == 'fit' or stage is None:
            self.train_ds = KcHousepricesDataset(X_train, y_train)
            self.val_ds = KcHousepricesDataset(X_val, y_val)
        if stage == 'test' or stage is None:
            self.test_ds = KcHousepricesDataset(X_test, y_test)
        
    def train_dataloader(self):
        if not self.train_ds:
            raise ValueError("Train dataset not setup. Call setup('fit') first.")
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    def val_dataloader(self):
        if not self.val_ds:
            raise ValueError("Validation dataset not setup. Call setup('fit') first.")
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))

    def test_dataloader(self):
        if not self.test_ds:
            raise ValueError("Test dataset not setup. Call setup('test') first.")
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=(self.num_workers > 0))
    
