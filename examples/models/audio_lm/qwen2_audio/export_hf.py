#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export a Megatron-Bridge checkpoint to HuggingFace format.

Usage:
    python export_hf.py \
        --megatron-path /path/to/checkpoints/iter_0001000 \
        --hf-path ./hf_exports/qwen2_audio_finetuned \
        --hf-model-path Qwen/Qwen2-Audio-7B-Instruct
"""

import argparse

from megatron.bridge import AutoBridge


def main():
    """Export Megatron checkpoint to HuggingFace format."""
    parser = argparse.ArgumentParser(description="Export Megatron checkpoint to HuggingFace format")
    parser.add_argument(
        "--megatron-path",
        type=str,
        required=True,
        help="Path to the Megatron checkpoint directory (e.g. nemo_experiments/.../checkpoints/iter_0001000)",
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="Output directory for HuggingFace model",
    )
    parser.add_argument(
        "--hf-model-path",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Original HuggingFace model ID (for config/tokenizer)",
    )
    args = parser.parse_args()

    print(f"Loading bridge from: {args.hf_model_path}")
    bridge = AutoBridge.from_hf_pretrained(args.hf_model_path)

    print(f"Exporting: {args.megatron_path} -> {args.hf_path}")
    bridge.export_ckpt(
        megatron_path=args.megatron_path,
        hf_path=args.hf_path,
    )
    print(f"Done! HuggingFace model saved to: {args.hf_path}")


if __name__ == "__main__":
    main()
