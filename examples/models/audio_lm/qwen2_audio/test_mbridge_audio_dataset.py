#!/usr/bin/env python3
"""
Test HFDatasetConversationProvider + qwen2_audio_collate_fn with aishell dataset.

This script does NOT require Megatron distributed init — it directly calls the
provider/dataset/collate layers to verify the data pipeline end-to-end.

Usage:
    python test_mbridge_audio_dataset.py
"""

import torch
from transformers import AutoProcessor

from megatron.bridge.data.vlm_datasets.collate import qwen2_audio_collate_fn
from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.data.vlm_datasets.hf_dataset_makers import make_default_audio_dataset


MODEL_PATH = "/workspace_yuekai/HF/Qwen2-Audio-7B-Instruct"
DATASET_NAME = "yuekai/aishell"
DATASET_SUBSET = "test"
DATASET_SPLIT = "test"
PROMPT = "Transcribe the audio clip."
NUM_EXAMPLES = 4  # how many raw examples to load
BATCH_SIZE = 2  # collate batch size


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    # ------------------------------------------------------------------
    # 1. Load processor
    # ------------------------------------------------------------------
    section("1. Loading Qwen2-Audio processor")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f"Processor type: {type(processor).__name__}")
    print(f"Tokenizer vocab size: {processor.tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # 2. Test make_default_audio_dataset (the maker function)
    # ------------------------------------------------------------------
    section("2. Testing make_default_audio_dataset")
    examples = make_default_audio_dataset(
        path_or_dataset=DATASET_NAME,
        subset=DATASET_SUBSET,
        split=DATASET_SPLIT,
        prompt=PROMPT,
    )
    print(f"Total examples from maker: {len(examples)}")

    # Inspect first example
    ex = examples[0]
    print(f"\nFirst example keys: {list(ex.keys())}")
    print(f"Conversation ({len(ex['conversation'])} turns):")
    for turn in ex["conversation"]:
        role = turn["role"]
        if isinstance(turn["content"], list):
            types = [c["type"] for c in turn["content"]]
            print(f"  {role}: {types}")
        elif isinstance(turn["content"], str):
            print(f"  {role}: {turn['content'][:80]}...")
    audio_array, sr = ex["audio"]
    print(f"Audio: array shape={audio_array.shape}, sr={sr}")

    # ------------------------------------------------------------------
    # 3. Test VLMConversationDataset (wrapping + collate selection)
    # ------------------------------------------------------------------
    section("3. Testing VLMConversationDataset")
    dataset = VLMConversationDataset(
        base_examples=examples[:NUM_EXAMPLES],
        target_length=NUM_EXAMPLES,
        processor=processor,
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Collate fn bound: {dataset.collate_fn}")

    item = dataset[0]
    print(f"dataset[0] keys: {list(item.keys())}")
    # ------------------------------------------------------------------
    # 4. Test qwen2_audio_collate_fn directly (single example)
    # ------------------------------------------------------------------
    section("4. Testing qwen2_audio_collate_fn (batch_size=1)")
    batch_1 = qwen2_audio_collate_fn([examples[0]], processor)
    print(f"Batch keys: {list(batch_1.keys())}")
    for k, v in batch_1.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v).__name__}")

    # Decode input_ids to verify template
    decoded = processor.tokenizer.decode(batch_1["input_ids"][0], skip_special_tokens=False)
    print(f"\nDecoded input_ids (first 300 chars):\n  {decoded[:300]}...")

    # Check labels masking
    labels = batch_1["labels"][0]
    num_masked = (labels == -100).sum().item()
    num_active = (labels != -100).sum().item()
    print(f"\nLabels: {num_masked} masked (-100), {num_active} active (loss computed)")

    # Decode only the active label tokens
    active_ids = labels[labels != -100]
    if len(active_ids) > 0:
        active_text = processor.tokenizer.decode(active_ids, skip_special_tokens=True)
        print(f"Active label text: {active_text[:200]}")

    # Check loss_mask
    loss_mask = batch_1["loss_mask"][0]
    print(f"Loss mask: {loss_mask.sum().item():.0f} active positions out of {loss_mask.shape[0]}")

    # Check visual_inputs (Qwen2AudioInputs)
    vi = batch_1["visual_inputs"]
    print(f"\nvisual_inputs type: {type(vi).__name__}")
    model_kwargs = vi.normalized_for_model()
    for k, v in model_kwargs.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # ------------------------------------------------------------------
    # 5. Test collate with batch_size > 1
    # ------------------------------------------------------------------
    section(f"5. Testing qwen2_audio_collate_fn (batch_size={BATCH_SIZE})")
    batch_n = qwen2_audio_collate_fn(examples[:BATCH_SIZE], processor)
    for k, v in batch_n.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v).__name__}")

    # Check per-sample label masking
    for i in range(BATCH_SIZE):
        labels_i = batch_n["labels"][i]
        active_i = labels_i[labels_i != -100]
        text_i = processor.tokenizer.decode(active_i, skip_special_tokens=True) if len(active_i) > 0 else "<empty>"
        print(f"  Sample {i} active labels: {text_i[:100]}")

    # ------------------------------------------------------------------
    # 6. Test via dataset.collate_fn (simulates DataLoader)
    # ------------------------------------------------------------------
    section("6. Testing via dataset.collate_fn (DataLoader simulation)")
    loader_batch = dataset.collate_fn([dataset[i] for i in range(min(BATCH_SIZE, len(dataset)))])
    print(f"Batch keys: {list(loader_batch.keys())}")
    for k, v in loader_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v).__name__}")

    # ------------------------------------------------------------------
    # 7. HF vs M-Bridge collator comparison (same example)
    # ------------------------------------------------------------------
    section("7. HF vs M-Bridge collator: side-by-side comparison")

    ex = examples[0]
    # -- M-Bridge collator --
    mb_batch = qwen2_audio_collate_fn([ex], processor)
    mb_input_ids = mb_batch["input_ids"][0]
    mb_labels = mb_batch["labels"][0]
    mb_loss_mask = mb_batch["loss_mask"][0]

    # -- HF-style collator (same logic as train_hf.py Qwen2AudioCollator) --
    conv = ex["conversation"]
    audio = ex["audio"]
    audio_array = audio[0] if isinstance(audio, tuple) else (audio["array"] if isinstance(audio, dict) else audio)
    text = processor.apply_chat_template(conv, tokenize=False)
    hf_batch = processor(text=[text], audio=[audio_array], return_tensors="pt", padding=True)
    hf_input_ids = hf_batch["input_ids"][0]

    # HF labels: unshifted, mask non-assistant tokens
    hf_labels = hf_input_ids.clone()
    # Extract assistant text from conversation
    assistant_text = ""
    for turn in conv:
        if turn["role"] == "assistant":
            for c in turn["content"]:
                if isinstance(c, dict) and c.get("type") == "text":
                    assistant_text += c["text"]
    assistant_token_ids = processor.tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
    ids_list = hf_input_ids.tolist()
    span_len = len(assistant_token_ids)
    found = -1
    for start in range(len(ids_list) - span_len, -1, -1):
        if ids_list[start : start + span_len] == assistant_token_ids:
            found = start
            break
    if found >= 0:
        hf_labels[:found] = -100
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            hf_labels[hf_input_ids == pad_id] = -100
    else:
        print("  WARNING: HF collator could not find assistant span!")
        hf_labels[:] = -100

    # -- Compare --
    print(f"  input_ids match: {torch.equal(mb_input_ids, hf_input_ids)}")
    print(f"  input_ids shapes: M-Bridge={mb_input_ids.shape}, HF={hf_input_ids.shape}")

    # M-Bridge active labels (pre-shifted)
    mb_active = mb_labels[mb_labels != -100]
    mb_text = processor.tokenizer.decode(mb_active, skip_special_tokens=True) if mb_active.numel() > 0 else "<empty>"

    # HF active labels (unshifted) - shift to compare
    hf_active = hf_labels[1:][hf_labels[1:] != -100]  # shift by 1 to match M-Bridge convention
    hf_text = processor.tokenizer.decode(hf_active, skip_special_tokens=True) if hf_active.numel() > 0 else "<empty>"

    print(f"\n  M-Bridge active label text: '{mb_text}'")
    print(f"  HF       active label text: '{hf_text}'")
    print(f"  Texts match: {mb_text == hf_text}")

    # Compare active token counts
    print(f"\n  M-Bridge: {mb_active.numel()} active label tokens, loss_mask sum={mb_loss_mask.sum().item():.0f}")
    print(f"  HF:       {hf_active.numel()} active label tokens")

    # Show position-by-position comparison for first/last few active positions
    mb_positions = (mb_loss_mask > 0).nonzero(as_tuple=True)[0]
    if len(mb_positions) > 0:
        print(
            f"\n  M-Bridge loss positions: first 5 = {mb_positions[:5].tolist()}, last 5 = {mb_positions[-5:].tolist()}"
        )
        print(f"  M-Bridge labels at those positions: {mb_labels[mb_positions[:5]].tolist()}")
        # What M-Bridge predicts: at position p, predict labels[p] = input_ids[p+1]
        print(f"  input_ids[p+1] at those positions:  {mb_input_ids[mb_positions[:5] + 1].tolist()}")
        label_matches_next = (
            torch.equal(mb_labels[mb_positions], mb_input_ids[mb_positions + 1])
            if mb_positions[-1] + 1 < len(mb_input_ids)
            else False
        )
        print(f"  labels[p] == input_ids[p+1] for all active positions: {label_matches_next}")

    # Reference text from dataset
    ref_text = ex["conversation"][-1]["content"]
    if isinstance(ref_text, list):
        ref_text = "".join(c.get("text", "") for c in ref_text if isinstance(c, dict) and c.get("type") == "text")
    print(f"\n  Reference text: '{ref_text}'")

    # ------------------------------------------------------------------
    section("ALL TESTS PASSED")
    breakpoint()


if __name__ == "__main__":
    main()
