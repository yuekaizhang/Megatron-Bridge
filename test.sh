export PYTHONPATH=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/pr/M-latest/src:$PYTHONPATH

# /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/pip install "qwen-omni-utils[decord]"
# apt-get update 2>&1 | tail -3 && apt-get install -y ffmpeg 2>&1
/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python -m torch.distributed.run --nproc_per_node=4 examples/conversion/hf_to_megatron_generate_omni_lm.py \
    --hf_model_path=Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --video_url="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/audio_visual.mp4" \
    --prompt="What was the first sentence the boy said when he met the girl?" \
    --use_audio_in_video \
    --tp 1 \
    --ep 4 \
    --etp 1 \
    --trust_remote_code
