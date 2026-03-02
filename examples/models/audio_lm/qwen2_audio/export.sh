export PYTHONPATH=/workspace_yuekai/asr/Megatron-Bridge:$PYTHONPATH
export TORCHDYNAMO_DISABLE=1

megatron_path=/workspace_yuekai/asr/Megatron-Bridge/examples/models/audio_lm/qwen2_audio/exp/qwen2_audio_7b_sft_tp1_pp1/iter_0003000
hf_path=/workspace_yuekai/asr/Megatron-Bridge/examples/models/audio_lm/qwen2_audio/exp/qwen2_audio_7b_sft_tp1_pp1/hf_iter_0003000

# megatron_path=/workspace_yuekai/asr/Megatron-Bridge/examples/models/alm/qwen2_audio/qwen2_audio_7b_instruct_2512/iter_0000000
# hf_path=examples/models/audio_lm/qwen2_audio/exp/qwen2_audio_7b_original
# hf_path=/workspace_yuekai/HF/Qwen2-Audio-7B-Instruct

uv run --no-sync python3 examples/models/audio_lm/qwen2_audio/export_hf.py \
  --megatron-path $megatron_path \
  --hf-model-path Qwen/Qwen2-Audio-7B-Instruct \
  --hf-path $hf_path

uv run --no-sync python3 examples/models/audio_lm/qwen2_audio/decode_hf.py \
    --hf-model-path $hf_path \
    --prompt "Detect the language and recognize the speech: <|zh|>"

# uv run --no-sync python3 examples/models/audio_lm/qwen2_audio/test_mbridge_audio_dataset.py
# uv run --no-sync torchrun --nproc_per_node=1 examples/models/audio_lm/qwen2_audio/debug_compare_forward.py 