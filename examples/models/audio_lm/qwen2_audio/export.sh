export PYTHONPATH=/workspace_yuekai/asr/Megatron-Bridge:$PYTHONPATH

megatron_path=/workspace/results/qwen2_audio_7b_sft_tp1_pp1/iter_0000200
# megatron_path=examples/models/audio_lm/qwen2_audio/qwen2_audio_7b_instruct_2512_bf16/iter_0000000
hf_path=examples/models/audio_lm/qwen2_audio/exp/qwen2_audio_7b_200q2

# uv run --no-sync python3 examples/models/audio_lm/qwen2_audio/export_hf.py \
#   --megatron-path $megatron_path \
#   --hf-model-path Qwen/Qwen2-Audio-7B-Instruct \
#   --hf-path $hf_path


# hf_path=/workspace_yuekai/HF/Qwen2-Audio-7B-Instruct
hf_path=/workspace_yuekai/asr/Megatron-Bridge/exp/qwen2_audio_aishell/checkpoint-100
# uv run --no-sync python3 examples/models/audio_lm/qwen2_audio/decode_hf.py \
#     --hf-model-path $hf_path


uv run --no-sync python3 examples/models/audio_lm/qwen2_audio/test_mbridge_audio_dataset.py