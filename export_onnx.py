# import torch
# import torch.onnx
# from pathlib import Path
# from tokenizers import Tokenizer

# from model import build_transformer
# from config import get_config

# def export_transformer_to_onnx():
#     # 1. Load configuration and specify languages
#     config = get_config()
#     config['batch_size'] = 1
#     config['lang_src'] = 'en'
#     config['lang_tgt'] = 'it'
    
#     # 2. Build the correct path to the tokenizer files in the root directory
#     tokenizer_file_pattern = config['tokenizer_file'] 
    
#     # Correctly point to the root project folder (no prefix needed)
#     tokenizer_src_path = "tokenizer_en.json"
#     tokenizer_tgt_path = "tokenizer_hi.json"

#     print(f"Attempting to load tokenizer from: {tokenizer_src_path}")
    
#     # Load tokenizers to get vocabulary sizes
#     tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
#     tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
#     config['src_vocab_size'] = tokenizer_src.get_vocab_size()
#     config['tgt_vocab_size'] = tokenizer_tgt.get_vocab_size()

#     # 3. Build the model
#     model = build_transformer(
#         config["src_vocab_size"],
#         config["tgt_vocab_size"],
#         config["seq_len"],
#         config["seq_len"],
#         d_model=config['d_model']
#     )

#     # 4. Load your trained model weights
#     model_path = r"C:\Users\NIMESH PARMAR\pytorch-transformer\pytorch-transformer\cfilt\iitb-english-hindi_weights\tmodel_25.pt" # <--- IMPORTANT: Update if needed
#     print(f"Loading model from {model_path}")
#     # state_dict = torch.load(model_path, map_location=torch.device('cpu'))
#     # model.load_state_dict(state_dict)
#     # Corrected code
#     checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     # 5. Create dummy inputs
#     seq_len = config['seq_len']
#     encoder_input = torch.randint(0, config['src_vocab_size'], (1, seq_len), dtype=torch.long)
#     decoder_input = torch.randint(0, config['tgt_vocab_size'], (1, seq_len), dtype=torch.long)
#     encoder_mask = (encoder_input != 0).unsqueeze(1).unsqueeze(2).int()
#     decoder_mask = (decoder_input != 0).unsqueeze(1).int() & torch.tril(torch.ones((1, seq_len, seq_len), dtype=torch.long))

#     # 6. Define the output file path
#     onnx_file_path = "transformer_model.onnx"

#     # 7. Export the model
#     print("Exporting model to ONNX...")
#     torch.onnx.export(
#         model,
#         (encoder_input, encoder_mask, decoder_input, decoder_mask),
#         onnx_file_path,
#         export_params=True,
#         opset_version=12,
#         do_constant_folding=True,
#         input_names=['encoder_input', 'encoder_mask', 'decoder_input', 'decoder_mask'],
#         output_names=['projection_output'],
#         dynamic_axes={
#             'encoder_input': {0: 'batch_size'},
#             'decoder_input': {0: 'batch_size'},
#             'projection_output': {0: 'batch_size'}
#         }
#     )
#     print(f"✅ Model successfully exported to {onnx_file_path}")

# if __name__ == '__main__':
#     export_transformer_to_onnx()
import torch
import torch.onnx
from pathlib import Path
from tokenizers import Tokenizer

from model import build_transformer
from config import get_config

def export_transformer_to_onnx():
    # 1. Load configuration and specify languages
    config = get_config()
    config['batch_size'] = 1
    config['lang_src'] = 'en'
    config['lang_tgt'] = 'hi' # Corrected language
    
    # 2. Correctly point to the tokenizer files
    tokenizer_src_path = "tokenizer_en.json"
    tokenizer_tgt_path = "tokenizer_hi.json"

    print(f"Attempting to load tokenizer from: {tokenizer_src_path}")
    
    # Load tokenizers to get vocabulary sizes
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
    config['src_vocab_size'] = tokenizer_src.get_vocab_size()
    config['tgt_vocab_size'] = tokenizer_tgt.get_vocab_size()

    # 3. Build the model
    model = build_transformer(
        config["src_vocab_size"],
        config["tgt_vocab_size"],
        config["seq_len"],
        config["seq_len"],
        d_model=config['d_model']
    )

    # 4. Load your trained model weights
    model_path = r"C:\Users\NIMESH PARMAR\pytorch-transformer\pytorch-transformer\cfilt\iitb-english-hindi_weights\tmodel_25.pt"
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 5. Create dummy inputs with CONSISTENT data types
    seq_len = config['seq_len']
    encoder_input = torch.randint(0, config['src_vocab_size'], (1, seq_len), dtype=torch.long)
    decoder_input = torch.randint(0, config['tgt_vocab_size'], (1, seq_len), dtype=torch.long)
    
    # --- THIS IS THE FIX ---
    # Change .int() to .long() to match the other inputs (int64)
    encoder_mask = (encoder_input != 0).unsqueeze(1).unsqueeze(2).long() 
    decoder_mask = (decoder_input != 0).unsqueeze(1).long() & torch.tril(torch.ones((1, seq_len, seq_len), dtype=torch.long))

    # 6. Define the output file path
    onnx_file_path = "transformer_model.onnx"

    # 7. Export the model
    print("Exporting model to ONNX...")
    torch.onnx.export(
        model,
        (encoder_input, encoder_mask, decoder_input, decoder_mask),
        onnx_file_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['encoder_input', 'encoder_mask', 'decoder_input', 'decoder_mask'],
        output_names=['projection_output'],
        dynamic_axes={
            'encoder_input': {0: 'batch_size'},
            'decoder_input': {0: 'batch_size'},
            'projection_output': {0: 'batch_size'}
        }
    )
    print(f"✅ Model successfully exported to {onnx_file_path}")

if __name__ == '__main__':
    export_transformer_to_onnx()
