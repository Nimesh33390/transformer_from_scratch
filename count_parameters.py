from pathlib import Path
from tokenizers import Tokenizer
import torch

from model import build_transformer
from config import get_config

def count_parameters():
    # Load configuration to build the model
    config = get_config()
    config['lang_src'] = 'en'
    config['lang_tgt'] = 'it'

    # --- Load tokenizers to get vocabulary sizes ---
    # This ensures the model is built with the correct dimensions
    try:
        tokenizer_file_pattern = config['tokenizer_file']
        tokenizer_src_path = "tokenizer_en.json"
        tokenizer_tgt_path = "tokenizer_hi.json"
        tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
        tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
        config['src_vocab_size'] = tokenizer_src.get_vocab_size()
        config['tgt_vocab_size'] = tokenizer_tgt.get_vocab_size()
    except FileNotFoundError:
        print("Tokenizer files not found in the root directory.")
        print("Please ensure 'tokenizer_en.json' and 'tokenizer_it.json' exist.")
        return

    # --- Build the Transformer model ---
    # Using the same function and parameters as your training script
    model = build_transformer(
        config["src_vocab_size"],
        config["tgt_vocab_size"],
        config["seq_len"],
        config["seq_len"],
        d_model=config['d_model']
    )

    # --- Count the parameters ---
    # This sums up the number of elements in each trainable weight/bias
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-" * 40)
    print(f"Transformer Model Parameter Count")
    print("-" * 40)
    print(f"Source Vocabulary Size: {config['src_vocab_size']}")
    print(f"Target Vocabulary Size: {config['tgt_vocab_size']}")
    print(f"Model Dimension (d_model): {config['d_model']}")
    print("-" * 40)
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"(Approximately {total_params / 1_000_000:.2f} Million)")
    print("-" * 40)

if __name__ == '__main__':
    count_parameters()