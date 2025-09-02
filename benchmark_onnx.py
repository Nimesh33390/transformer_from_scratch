import time
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path

def benchmark_provider(model_path, provider, src_vocab_size, tgt_vocab_size):
    """
    Measures the inference latency of an ONNX model on a specific provider (CPU or GPU).
    """
    assert provider in ort.get_available_providers(), f"Provider {provider} not available."
    
    # Create an inference session with the specified provider
    session = ort.InferenceSession(model_path, providers=[provider])
    
    # Get the expected input shapes and names
    input_meta = session.get_inputs()
    encoder_input_name = input_meta[0].name
    encoder_mask_name = input_meta[1].name
    decoder_input_name = input_meta[2].name
    decoder_mask_name = input_meta[3].name
    
    seq_len = input_meta[0].shape[1]
    
    # --- THIS IS THE FIX ---
    # Use the REAL vocabulary sizes to generate valid random token IDs
    encoder_input = np.random.randint(0, src_vocab_size, (1, seq_len), dtype=np.int64)
    decoder_input = np.random.randint(0, tgt_vocab_size, (1, seq_len), dtype=np.int64)
    
    encoder_mask = np.ones((1, 1, 1, seq_len), dtype=np.int64)
    decoder_mask = np.ones((1, seq_len, seq_len), dtype=np.int64)

    inputs = {
        encoder_input_name: encoder_input,
        encoder_mask_name: encoder_mask,
        decoder_input_name: decoder_input,
        decoder_mask_name: decoder_mask
    }

    # Warm-up run
    session.run(None, inputs)

    # Timed run
    num_runs = 100
    start_time = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, inputs)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_runs) * 1000
    
    return avg_latency_ms

if __name__ == '__main__':
    onnx_model_path = "transformer_model.onnx"

    if not os.path.exists(onnx_model_path):
        print(f"Error: ONNX model file not found at '{onnx_model_path}'")
    else:
        # --- Load tokenizers to get accurate vocabulary sizes ---
        try:
            tokenizer_src = Tokenizer.from_file(str(Path("tokenizer_en.json")))
            tokenizer_tgt = Tokenizer.from_file(str(Path("tokenizer_hi.json")))
            src_vocab_size = tokenizer_src.get_vocab_size()
            tgt_vocab_size = tokenizer_tgt.get_vocab_size()
        except FileNotFoundError:
            print("Error: Tokenizer files not found. Please ensure 'tokenizer_en.json' and 'tokenizer_hi.json' are in the same directory.")
            exit()

        # --- Benchmark on CPU ---
        print("-" * 40)
        print("Running benchmark on CPU...")
        cpu_latency = benchmark_provider(onnx_model_path, "CPUExecutionProvider", src_vocab_size, tgt_vocab_size)
        print(f"Average CPU Latency: {cpu_latency:.2f} ms per sentence")
        print(f"Sentences per second (Throughput): {1000 / cpu_latency:.2f}")
        print("-" * 40)
        
        # --- Benchmark on GPU ---
        if "CUDAExecutionProvider" in ort.get_available_providers():
            print("Running benchmark on GPU (CUDA)...")
            gpu_latency = benchmark_provider(onnx_model_path, "CUDAExecutionProvider", src_vocab_size, tgt_vocab_size)
            print(f"Average GPU Latency: {gpu_latency:.2f} ms per sentence")
            print(f"Sentences per second (Throughput): {1000 / gpu_latency:.2f}")
            print("-" * 40)
            
            if cpu_latency > 0:
                print(f"GPU is approximately {cpu_latency / gpu_latency:.1f}x faster than CPU.")
                print("-" * 40)
        else:
            print("CUDAExecutionProvider not found. Skipping GPU benchmark.")
            print("-" * 40)

