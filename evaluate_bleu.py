import torch
from pathlib import Path
from tqdm import tqdm
import torchmetrics

from model import build_transformer
from config import get_config
from dataset import causal_mask
from train import get_ds  # This function is defined in your train.py, which imports it from dataset.py
from tokenizers import Tokenizer

def calculate_bleu_score():
    # --- 1. SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config = get_config()
    config['batch_size'] = 1 # Process one sentence at a time

    # --- 2. LOAD TOKENIZERS AND DATASET ---
    try:
        tokenizer_src_path = "tokenizer_en.json"
        tokenizer_tgt_path = "tokenizer_hi.json"
        tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
        tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
    except FileNotFoundError:
        print("Tokenizer files not found in the root directory.")
        print(f"Please ensure '{config['tokenizer_file'].format(config['lang_src'])}' and '{config['tokenizer_file'].format(config['lang_tgt'])}' exist.")
        return

    # Load validation dataset - THIS LINE IS NOW FIXED
    _, val_dataloader, _, _ = get_ds(config)

    # --- 3. BUILD AND LOAD THE TRAINED MODEL ---
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config['d_model']
    ).to(device)

    # Load the trained weights
    model_filename = "cfilt/iitb-english-hindi_weights/tmodel_25.pt" # <--- UPDATE THIS PATH
    print(f"Loading model: {model_filename}")
    
    try:
        checkpoint = torch.load(model_filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print(f"Model file not found at {model_filename}")
        return
    except KeyError:
        print("Could not find 'model_state_dict' in the checkpoint. Ensure the checkpoint file is correct.")
        return

    model.eval()

    # --- 4. EVALUATION LOOP ---
    source_texts = []
    expected_texts = []
    predicted_texts = []

    # Get the special tokens
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            # Assert that batch size is 1 for greedy decoding
            assert encoder_input.size(0) == 1, "Batch size must be 1 for evaluation."

            # --- Greedy Decoding ---
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
            
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            source_texts.append(source_text)
            expected_texts.append(target_text)
            predicted_texts.append(model_out)

    # --- 5. CALCULATE BLEU SCORE ---
    metric = torchmetrics.BLEUScore()
    # bleu = metric(predicted_texts, [expected_texts]) # BLEUScore expects a list of list of references
    bleu = metric(predicted_texts, [[text] for text in expected_texts])
    print("-" * 40)
    print("BLEU Score Evaluation Results")
    print("-" * 40)
    print(f"Model: {model_filename}")
    print(f"Total Sentences Evaluated: {len(expected_texts)}")
    print(f"BLEU Score: {bleu.item() * 100:.2f}")
    print("-" * 40)

    # Print a few examples
    print("Example Translations:")
    for i in range(min(3, len(expected_texts))):
        print(f"  Source:   {source_texts[i]}")
        print(f"  Expected: {expected_texts[i]}")
        print(f"  Predicted: {predicted_texts[i]}")
        print()


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return tokenizer_tgt.decode(decoder_input[0].tolist())

if __name__ == '__main__':
    calculate_bleu_score()

