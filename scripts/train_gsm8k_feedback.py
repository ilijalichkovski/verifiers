"""
Quick training script to test TextFeedbackRLTrainer on GSM8K.

Usage:
1. Start vLLM server:
   CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model Qwen/Qwen2.5-0.5B-Instruct --enforce-eager

2. Run training:
   CUDA_VISIBLE_DEVICES=1 uv run accelerate launch --num-processes 1 \
       --config-file configs/zero3.yaml scripts/train_gsm8k_feedback.py
"""

import verifiers as vf
from verifiers.rl.trainer import TextFeedbackRLTrainer, RLConfig

# Load GSM8K environment
env = vf.load_environment(env_id="gsm8k")

# Create trainer with textual feedback support
trainer = TextFeedbackRLTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    env=env,
    args=RLConfig(
        run_name="gsm8k-feedback-test",
        batch_size=16,
        micro_batch_size=4,
        rollouts_per_example=4,
        max_steps=10,
        max_seq_len=1024,
        log_completions_every_n_steps=1,  # Log every step to see feedback
    ),
)

trainer.train()

