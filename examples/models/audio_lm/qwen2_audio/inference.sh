#!/bin/bash
# Qwen2-Audio Megatron Bridge Inference Test
#
# This script demonstrates how to run inference with Qwen2-Audio models
# using Megatron-Bridge.
#
# Prerequisites:
# - librosa: pip install librosa
#
# Usage:
#   bash examples/models/audio_lm/qwen2_audio/inference.sh

set -e
export PYTHONPATH=/workspace_yuekai/asr/Megatron-Bridge:$PYTHONPATH
export WORKSPACE=${WORKSPACE:-/workspace}
export HF_MODEL="/workspace_yuekai/HF/Qwen2-Audio-7B-Instruct"
export MEGATRON_PATH="/workspace_yuekai/asr/Megatron-Bridge/examples/models/alm/qwen2_audio/qwen2_audio_7b_instruct_2512"

# Audio sample URLs from Qwen2-Audio demo
AUDIO_URL_1="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
AUDIO_URL_2="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
AUDIO_URL_3="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"

echo "============================================"
echo "Qwen2-Audio Megatron Bridge Inference Test"
echo "============================================"

# Option 1: Direct inference from HuggingFace (no conversion)
echo ""
echo "Option 1: Direct inference from HuggingFace..."
echo "Audio: ${AUDIO_URL_1}"
echo ""

uv run --no-sync python -m torch.distributed.run --nproc_per_node=1 examples/conversion/hf_to_megatron_generate_audio_lm.py \
  --hf_model_path ${HF_MODEL} \
  --audio_url "${AUDIO_URL_1}" \
  --prompt "What's that sound?" \
  --tp 1 \
  --max_new_tokens 50

# Option 2: Convert to Megatron format and run inference
# Uncomment the following to test checkpoint conversion workflow

echo ""
echo "Option 2: Converting HF checkpoint to Megatron format..."
# uv run python examples/conversion/convert_checkpoints.py import \
#   --hf-model ${HF_MODEL} \
#   --megatron-path ${MEGATRON_PATH}

# echo ""
# echo "Running inference on converted checkpoint..."
# uv run python -m torch.distributed.run examples/conversion/hf_to_megatron_generate_audio_lm.py \
#   --hf_model_path ${HF_MODEL} \
#   --megatron_model_path ${MEGATRON_PATH}/iter_0000000 \
#   --audio_url "${AUDIO_URL_1}" \
#   --prompt "What's that sound?" \
#   --max_new_tokens 50

echo ""
echo "============================================"
echo "Inference complete!"
echo "============================================"
