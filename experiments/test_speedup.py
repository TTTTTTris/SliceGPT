import torch
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def benchmark_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16,
        device_map="auto",
    )
    if args.sliced_model_path:
        model = torch.load(args.sliced_model_path).to(device)
    model.eval()
    print(model)
    
    # Prepare input batch
    input_text = "Hello, how are you?"
    inputs = tokenizer(
        [input_text] * args.batch_size,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.prefill_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    print("Running warmup...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=args.generate_tokens)
    torch.cuda.synchronize()

    # Prefill stage: process the prompt
    print("Benchmarking...")
    with torch.no_grad():
        start_time = time.time()

        outputs = model(
            **inputs,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values  # KV cache for fast decoding

        # Now fast autoregressive decoding using cache
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy decode first token
        generated_tokens = [next_token]

        for _ in range(args.generate_tokens - 1):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)

        torch.cuda.synchronize()
        end_time = time.time()

    elapsed_time = end_time - start_time
    total_tokens = args.batch_size * (args.prefill_length + args.generate_tokens)
    tokens_per_second = total_tokens / elapsed_time

    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLaMA model inference speed.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path.")
    parser.add_argument("--sliced_model_path", type=str, help="Path to the local model.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--prefill_length", type=int, default=512, help="Length of input (prompt) tokens.")
    parser.add_argument("--generate_tokens", type=int, default=128, help="Number of tokens to generate.")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="fp16", help="Data type: fp16 or bf16.")

    args = parser.parse_args()
    benchmark_inference(args)
