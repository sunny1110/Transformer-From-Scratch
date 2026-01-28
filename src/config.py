from pathlib import Path

def get_config():
    
    return {
        "batch_size": 32,
        "num_epochs": 1000,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "N": 6, # Defaulted to 6
        "h": 8, # Defaulted to 8
        "dropout": 0.1,
        "d_ff": 2048,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{}.json",
        "experiment_name": "runs/tmodel",
        "overfit_run": False
    }

def get_weights_file_path(config, epoch: str) -> str:

    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.')/ model_folder/ model_filename)