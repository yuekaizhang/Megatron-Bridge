# Qwen2-Audio Megatron Bridge

This directory contains examples for using Qwen2-Audio models with Megatron-Bridge.

## Overview

Qwen2-Audio is an audio-language model that combines:
- **Audio Encoder**: A Whisper-like encoder for processing mel spectrograms
- **Multimodal Projector**: Projects audio features to language model space
- **Language Model**: Qwen2-based LLM for text generation

## Supported Models

- [Qwen/Qwen2-Audio-7B](https://huggingface.co/Qwen/Qwen2-Audio-7B)
- [Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)

## Prerequisites

```bash
# Install required dependencies
pip install librosa transformers>=4.40.0
```

## Quick Start

### 1. Direct Inference from HuggingFace

```bash
# Run inference with audio from URL
python examples/conversion/hf_to_megatron_generate_alm.py \
  --hf_model_path "Qwen/Qwen2-Audio-7B-Instruct" \
  --audio_url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3" \
  --prompt "What's that sound?" \
  --max_new_tokens 50
```

### 2. Convert to Megatron Format

```bash
# Convert HuggingFace checkpoint to Megatron format
python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen2-Audio-7B-Instruct \
  --megatron-path /workspace/models/Qwen2-Audio-7B-Instruct
```

### 3. Run Inference from Megatron Checkpoint

```bash
# Run inference from converted Megatron checkpoint
python examples/conversion/hf_to_megatron_generate_alm.py \
  --hf_model_path "Qwen/Qwen2-Audio-7B-Instruct" \
  --megatron_model_path /workspace/models/Qwen2-Audio-7B-Instruct/iter_0000000 \
  --audio_url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3" \
  --prompt "What's that sound?" \
  --max_new_tokens 50
```

## Using the Inference Script

```bash
# Run the provided inference script
bash examples/models/alm/qwen2_audio/inference.sh
```

## API Usage

```python
from megatron.bridge import AutoBridge
from megatron.bridge.models import (
    Qwen2AudioModel,
    Qwen2AudioModelProvider,
    Qwen2AudioBridge,
)

# Load from HuggingFace
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

# Convert to Megatron provider
provider = bridge.to_megatron_provider()

# Get the model
model = provider.provide()
```

## Model Architecture

```
Qwen2AudioForConditionalGeneration
├── audio_tower (Qwen2AudioEncoder)
│   ├── conv1
│   ├── conv2
│   ├── embed_positions
│   ├── layers (32x Qwen2AudioEncoderLayer)
│   ├── layer_norm
│   └── avg_pooler
├── multi_modal_projector (Qwen2AudioMultiModalProjector)
│   └── linear (audio_dim -> text_dim)
└── language_model (Qwen2ForCausalLM)
    ├── model.embed_tokens
    ├── model.layers (28x decoder layers)
    ├── model.norm
    └── lm_head
```

## Weight Mappings

| Megatron Path | HuggingFace Path |
|---------------|------------------|
| `audio_tower.**` | `audio_tower.**` (replicated) |
| `multi_modal_projector.**` | `multi_modal_projector.**` (replicated) |
| `language_model.embedding.word_embeddings.weight` | `language_model.model.embed_tokens.weight` |
| `language_model.decoder.layers.*.self_attention.linear_qkv.weight` | `language_model.model.layers.*.self_attn.{q,k,v}_proj.weight` (merged) |
| `language_model.decoder.layers.*.mlp.linear_fc1.weight` | `language_model.model.layers.*.mlp.{gate,up}_proj.weight` (merged) |
| `language_model.output_layer.weight` | `language_model.lm_head.weight` |

## Audio Input Format

- **Input**: Mel spectrogram of shape `(batch, num_mel_bins, seq_len)`
- **num_mel_bins**: 128 (default)
- **Sampling Rate**: 16000 Hz
- **Feature Attention Mask**: Handles variable-length audio

## Sample Audio URLs

```python
# Glass breaking sound
audio_url_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"

# Throat clearing sound
audio_url_2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"

# Speech sample
audio_url_3 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"
```

## Expected Outputs

| Audio | Prompt | Expected Response |
|-------|--------|-------------------|
| glass-breaking | "What's that sound?" | "It is the sound of glass shattering." |
| throat-clearing | "What can you hear?" | "I can hear someone clearing their throat." |
| speech sample | "What does the person say?" | [Transcription of the speech] |

## Troubleshooting

### ImportError: librosa not found
```bash
pip install librosa
```

### ImportError: Qwen2Audio model requires transformers >= 4.40.0
```bash
pip install 'transformers>=4.40.0'
```

### CUDA out of memory
- Use tensor parallelism: `--tp 2` or `--tp 4`
- Use pipeline parallelism: `--pp 2`

## References

- [Qwen2-Audio Paper](https://arxiv.org/abs/2407.10759)
- [Qwen2-Audio HuggingFace](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
- [Qwen2-Audio GitHub](https://github.com/QwenLM/Qwen2-Audio)
