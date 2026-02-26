#!/usr/bin/env python3
"""
Decode an audio dataset with Qwen2-Audio for ASR evaluation.

Usage:
    python test_hf_audio_data.py \
        --hf-model-path /workspace_yuekai/HF/Qwen2-Audio-7B-Instruct \
        --dataset yuekai/aishell \
        --subset test \
        --split test \
        --prompt "Transcribe the audio clip." \
        --batch-size 4 \
        --max-samples 100 \
        --output results.jsonl
"""

import argparse
import json
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


def parse_args():
    """Parse command-line arguments for ASR decoding."""
    parser = argparse.ArgumentParser(description="ASR decoding with Qwen2-Audio on HF datasets")
    parser.add_argument("--hf-model-path", type=str, default="examples/models/audio_lm/qwen2_audio/exp/qwen2_audio_7b")
    # parser.add_argument("--hf-model-path", type=str, default="/workspace_yuekai/HF/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="yuekai/aishell", help="HuggingFace dataset name or path")
    parser.add_argument("--subset", type=str, default="test", help="Dataset subset/config name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--audio-column", type=str, default="audio", help="Column name for audio data")
    parser.add_argument("--text-column", type=str, default="text", help="Column name for reference text")
    parser.add_argument("--prompt", type=str, default="Transcribe the audio clip.", help="Prompt for the model")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max-samples", type=int, default=2, help="Max number of samples to decode (None=all)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file path (default: print to stdout)")
    return parser.parse_args()


def build_conversation(prompt):
    """Build a single-turn conversation for ASR."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def decode_batch(model, processor, batch_audio, prompt, max_new_tokens):
    """Run inference on a batch of audio samples."""
    conversations = [build_conversation(prompt) for _ in batch_audio]
    texts = [processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False) for conv in conversations]

    audios = []
    for audio in batch_audio:
        if isinstance(audio, dict):
            audios.append(audio["array"])
        elif isinstance(audio, tuple):
            audios.append(audio[0])
        else:
            audios.append(audio)

    inputs = processor(text=texts, audio=audios, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Strip the prompt tokens
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

    responses = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return responses


def main():
    """Run ASR decoding with Qwen2-Audio."""
    args = parse_args()

    print(f"Loading model: {args.hf_model_path}")
    processor = AutoProcessor.from_pretrained(args.hf_model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.hf_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    print(f"Loading dataset: {args.dataset} (subset={args.subset}, split={args.split})")
    dataset = load_dataset(args.dataset, args.subset, split=args.split)

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Decoding {len(dataset)} samples with batch_size={args.batch_size}")

    results = []
    out_file = open(args.output, "w", encoding="utf-8") if args.output else None
    t0 = time.time()

    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Decoding"):
        batch = dataset[i : i + args.batch_size]
        batch_audio = batch[args.audio_column]
        batch_ref = batch.get(args.text_column, [None] * len(batch_audio))

        predictions = decode_batch(model, processor, batch_audio, args.prompt, args.max_new_tokens)

        for j, (pred, ref) in enumerate(zip(predictions, batch_ref)):
            entry = {
                "id": i + j,
                "prediction": pred.strip(),
                "reference": ref.strip() if ref else "",
            }
            results.append(entry)
            if out_file:
                out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0

    if out_file:
        out_file.close()
        print(f"Results saved to: {args.output}")

    # Print summary
    print(f"\nDecoded {len(results)} samples in {elapsed:.1f}s ({len(results) / elapsed:.1f} samples/s)")

    # Print a few examples
    n_show = min(5, len(results))
    print(f"\nFirst {n_show} examples:")
    for r in results[:n_show]:
        print(f"  [{r['id']}] REF: {r['reference']}")
        print(f"       HYP: {r['prediction']}")
        print()


if __name__ == "__main__":
    main()
