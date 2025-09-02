from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

# --- We copy the BilingualDataset class from the original dataset.py ---
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # This is where the error was happening. tokenizer_tgt needs to be a Tokenizer object.
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [self.sos_token, torch.tensor(enc_input_tokens, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)],
            dim=0,
        )
        decoder_input = torch.cat(
            [self.sos_token, torch.tensor(dec_input_tokens, dtype=torch.int64), torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)],
            dim=0,
        )
        label = torch.cat(
            [torch.tensor(dec_input_tokens, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)],
            dim=0,
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

# --- This function ensures the Tokenizer OBJECTS are loaded and passed ---
def get_test_ds(config):
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Load existing tokenizers from file paths
    tokenizer_src_path = "tokenizer_en.json"
    tokenizer_tgt_path = "tokenizer_hi.json"
    
    tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
    tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

    # Select the specified slice of the data for testing
    start_index = config['test_slice_start']
    end_index = config['test_slice_end']
    print(f"Loading test data from row {start_index} to {end_index}...")
    test_ds_raw = ds_raw.select(range(start_index, end_index))
    
    # Create the dataset and dataloader
    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return test_dataloader, tokenizer_src, tokenizer_tgt

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
