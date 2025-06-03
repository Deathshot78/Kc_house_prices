import pandas as pd
import matplotlib.pyplot as plt
import os
import glob # Keep for flexibility, though direct path construction is prioritized

# --- Configuration: Set these paths based on your log structure ---

# METHOD 1: (RECOMMENDED FOR SIMPLICITY IF YOU KNOW THE RUN)
# Set the path to the specific main run directory you want to plot.
# This is the folder that CONTAINS 'train_logs', 'model_checkpoints', etc.
# Ensure this path is correct relative to where you run the plotting script, or use an absolute path.
MAIN_RUN_DIR_TO_PLOT = "projects/kc_house_prices/logs/lightning_logs/kc_house_mlp_final_run_20250603_014431"

TRAINING_LOGS_SUBDIR_NAME = "train_logs"
# This should match the SEED used in your train_final_model.py script for versioning the CSVLogger
SEED_VERSION_NAME = "seed42" 

# METHOD 2: (ALTERNATIVE - DIRECT PATH TO CSV)
# If you want to bypass the find function and point directly to the CSV.
# Uncomment and use this if Method 1 (using MAIN_RUN_DIR_TO_PLOT) gives trouble.
# DIRECT_METRICS_CSV_PATH = "projects/kc_house_prices/logs/lightning_logs/kc_house_mlp_final_run_20250603_014431/train_logs/seed42/metrics.csv"
DIRECT_METRICS_CSV_PATH = None # Set to None to use Method 1 by default


def find_training_metrics_csv(main_run_directory_path: str, 
                              training_logs_subdir: str, 
                              version_folder_name: str):
    """
    Constructs the direct path to the metrics.csv file based on the known structure.
    """
    expected_metrics_path = os.path.join(
        main_run_directory_path, 
        training_logs_subdir, 
        version_folder_name, 
        "metrics.csv"
    )

    if os.path.exists(expected_metrics_path):
        print(f"Found metrics file: {expected_metrics_path}")
        return expected_metrics_path
    else:
        print(f"Metrics file not found at expected path: {expected_metrics_path}")
        print("Please verify the following paths and names in your plotting script:")
        print(f"  - MAIN_RUN_DIR_TO_PLOT: '{main_run_directory_path}'")
        print(f"  - TRAINING_LOGS_SUBDIR_NAME: '{training_logs_subdir}'")
        print(f"  - SEED_VERSION_NAME (folder name): '{version_folder_name}'")
        return None

def plot_training_validation_metrics(metrics_file_path):
    if metrics_file_path is None or not os.path.exists(metrics_file_path):
        print(f"Metrics file not found or path is None. Cannot generate plots.")
        return

    try:
        metrics_df = pd.read_csv(metrics_file_path)
    except Exception as e:
        print(f"Error loading metrics.csv: {e}")
        return

    print("\nAvailable columns in metrics.csv for plotting:")
    print(metrics_df.columns.tolist())
    
    if 'epoch' not in metrics_df.columns:
        print("Error: 'epoch' column not found. Cannot proceed.")
        return
    
    x_axis_col = 'epoch'

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        print("Style 'seaborn-v0_8-darkgrid' not found. Trying 'ggplot'.")
        try:
            plt.style.use('ggplot')
        except OSError:
            print("Style 'ggplot' not found. Using Matplotlib default style with grid.")

    train_loss_col = 'train_loss_epoch' 
    val_loss_col = 'val_loss'          
    train_mape_log_col = 'train_mape_log'
    val_mape_log_col = 'val_mape_log'    
    train_mae_log_col = 'train_mae_log'  
    val_mae_log_col = 'val_mae_log'      
    val_mape_dollars_col = 'val_mape_dollars'
    val_mae_dollars_col = 'val_mae_dollars'  
    lr_col = next((col for col in metrics_df.columns if col.startswith('lr-')), None)

    # Plotting functions (Loss, MAPE log, MAE log, LR)
    # Example for Loss:
    if train_loss_col in metrics_df.columns and val_loss_col in metrics_df.columns:
        plot_df_train_loss = metrics_df[[x_axis_col, train_loss_col]].dropna()
        plot_df_val_loss = metrics_df[[x_axis_col, val_loss_col]].dropna()
        
        if not plot_df_train_loss.empty and not plot_df_val_loss.empty:
            plt.figure(figsize=(12, 7))
            plt.plot(plot_df_train_loss[x_axis_col], plot_df_train_loss[train_loss_col], label='Train Loss (Log-space)', linestyle='-')
            plt.plot(plot_df_val_loss[x_axis_col], plot_df_val_loss[val_loss_col], label='Validation Loss (Log-space)', linestyle='--')
            plt.xlabel(x_axis_col.capitalize()); plt.ylabel('Loss (Huber, Log-space)')
            plt.title('Training & Validation Loss (Log-Space) vs. Epoch'); plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig("loss_log_space_plot.png"); plt.show()
        else: print(f"Not enough data for loss plot.")
    else: print(f"Loss columns not found: train='{train_loss_col}', val='{val_loss_col}'")

    # Add other plots similarly (MAPE log, MAE log, lr)
    # --- Plot MAPE (Log-Space) ---
    if train_mape_log_col in metrics_df.columns and val_mape_log_col in metrics_df.columns:
        plot_df_train_mape = metrics_df[[x_axis_col, train_mape_log_col]].dropna()
        plot_df_val_mape = metrics_df[[x_axis_col, val_mape_log_col]].dropna()
        if not plot_df_train_mape.empty and not plot_df_val_mape.empty:
            plt.figure(figsize=(12, 7))
            plt.plot(plot_df_train_mape[x_axis_col], plot_df_train_mape[train_mape_log_col], label='Train MAPE (Log-space %)', linestyle='-')
            plt.plot(plot_df_val_mape[x_axis_col], plot_df_val_mape[val_mape_log_col], label='Validation MAPE (Log-space %)', linestyle='--')
            plt.xlabel(x_axis_col.capitalize()); plt.ylabel('MAPE (Log-space %)')
            plt.title('Training & Validation MAPE (Log-Space) vs. Epoch'); plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig("mape_log_space_plot.png"); plt.show()
        else: print(f"Not enough data for log-space MAPE plot.")
    else: print(f"Log-space MAPE columns not found: train='{train_mape_log_col}', val='{val_mape_log_col}'")

    # --- Plot MAE (Log-Space) ---
    if train_mae_log_col in metrics_df.columns and val_mae_log_col in metrics_df.columns:
        plot_df_train_mae = metrics_df[[x_axis_col, train_mae_log_col]].dropna()
        plot_df_val_mae = metrics_df[[x_axis_col, val_mae_log_col]].dropna()
        if not plot_df_train_mae.empty and not plot_df_val_mae.empty:
            plt.figure(figsize=(12, 7))
            plt.plot(plot_df_train_mae[x_axis_col], plot_df_train_mae[train_mae_log_col], label='Train MAE (Log-space)', linestyle='-')
            plt.plot(plot_df_val_mae[x_axis_col], plot_df_val_mae[val_mae_log_col], label='Validation MAE (Log-space)', linestyle='--')
            plt.xlabel(x_axis_col.capitalize()); plt.ylabel('MAE (Log-space)')
            plt.title('Training & Validation MAE (Log-Space) vs. Epoch'); plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig("mae_log_space_plot.png"); plt.show()
        else: print(f"Not enough data for log-space MAE plot.")
    else: print(f"Log-space MAE columns not found: train='{train_mae_log_col}', val='{val_mae_log_col}'")
        
    # --- Plot Learning Rate ---
    if lr_col and lr_col in metrics_df.columns :
        epoch_aligned_lr_df = metrics_df[[x_axis_col, lr_col]].dropna()
        if not epoch_aligned_lr_df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(epoch_aligned_lr_df[x_axis_col], epoch_aligned_lr_df[lr_col], label='Learning Rate', marker='.', linestyle='-')
            plt.xlabel(x_axis_col.capitalize()); plt.ylabel('Learning Rate')
            plt.title('Learning Rate vs. ' + x_axis_col.capitalize()); plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig("lr_plot.png"); plt.show()
        else: 
            print(f"\nNo direct epoch-aligned LR data for '{lr_col}'. Attempting step-based plot.")
            if 'step' in metrics_df.columns:
                step_lr_df = metrics_df[metrics_df[lr_col].notna()][['step', lr_col]]
                if not step_lr_df.empty:
                    plt.figure(figsize=(10, 6))
                    plt.plot(step_lr_df['step'], step_lr_df[lr_col], label='Learning Rate (vs. Step)', marker='.', linestyle='-')
                    plt.xlabel('Step'); plt.ylabel('Learning Rate'); plt.title('Learning Rate vs. Step')
                    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("lr_vs_step_plot.png"); plt.show()
                else: print(f"No non-NaN LR values with 'step' for '{lr_col}'.")
            else: print("The 'step' column not available for fallback LR plot.")
    else: print(f"Learning rate column (starting with 'lr-') not found: '{lr_col}'.")


if __name__ == "__main__":
    metrics_file_to_plot = None

    if DIRECT_METRICS_CSV_PATH and os.path.exists(DIRECT_METRICS_CSV_PATH):
        print(f"Using direct path to metrics.csv: {DIRECT_METRICS_CSV_PATH}")
        metrics_file_to_plot = DIRECT_METRICS_CSV_PATH
    elif MAIN_RUN_DIR_TO_PLOT:
        print(f"Attempting to find metrics using MAIN_RUN_DIR_TO_PLOT: {MAIN_RUN_DIR_TO_PLOT}")
        metrics_file_to_plot = find_training_metrics_csv(
            main_run_directory_path=MAIN_RUN_DIR_TO_PLOT,
            training_logs_subdir=TRAINING_LOGS_SUBDIR_NAME,
            version_folder_name=SEED_VERSION_NAME 
        )
    else:
        print("Please set either 'DIRECT_METRICS_CSV_PATH' or 'MAIN_RUN_DIR_TO_PLOT' at the top of the script.")

    if metrics_file_to_plot:
        plot_training_validation_metrics(metrics_file_to_plot)
    else:
        print("Could not find metrics.csv to plot. Please check your path configurations.")