from pathlib import Path

def get_config():
    return {
        "batch_size": 1,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 100,
        "d_model": 512,
        "datasource": 'cfilt/iitb-english-hindi',
        "lang_src": "en",
        "lang_tgt": "hi", # <-- THIS VALUE IS NOW CORRECT (changed from "it")
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "test_slice_start": 25000,
        "test_slice_end": 27500
    }

