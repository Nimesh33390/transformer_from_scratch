import torch
from pathlib import Path
from tqdm import tqdm
import torchmetrics

# --- Import from your NEW, separate files ---
from model import build_transformer
from config_test import get_config # <-- Uses config_test.py
from dataset_test import get_test_ds, causal_mask # <-- Uses dataset_test.py

def calculate_bleu_score():
    # --- 1. SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config = get_config()
    config['batch_size'] = 1 # Process one sentence at a time

    # --- 2. LOAD TOKENIZERS AND TEST DATASET ---
    test_dataloader, tokenizer_src, tokenizer_tgt = get_test_ds(config)

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
    
    checkpoint = torch.load(model_filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- 4. EVALUATION LOOP ---
    source_texts = []
    expected_texts = []
    predicted_texts = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Running Final Test"):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
            
            source_texts.append(batch["src_text"][0])
            expected_texts.append(batch["tgt_text"][0])
            predicted_texts.append(model_out)

    # --- 5. CALCULATE FINAL BLEU SCORE ---
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted_texts, [[text] for text in expected_texts])

    print("-" * 40)
    print("Final BLEU Score Test Results")
    print("-" * 40)
    print(f"Model: {model_filename}")
    print(f"Test Sentences Evaluated: {len(expected_texts)}")
    print(f"Final BLEU Score: {bleu.item() * 100:.2f}")
    print("-" * 40)

    # Print a few examples
    print("Example Translations from Test Set:")
    for i in range(min(5, len(expected_texts))):
        print(f"  Source:   {source_texts[i]}")
        print(f"  Expected: {expected_texts[i]}")
        print(f"  Predicted: {predicted_texts[i]}")
        print()


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
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