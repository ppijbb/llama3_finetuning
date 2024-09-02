import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer  # DPOTrainer 사용
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import deepspeed
from data_module import get_dataset

# 모델과 토크나이저 불러오기
model_id = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id,
                                            device_map="auto",
                                            torch_dtype=torch.bfloat16,
                                            trust_remote_code=True,
                                            return_dict=True)
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          trust_remote_code=True,
                                          use_fast=True)

# Lora 설정 정의
lora_config = LoraConfig(
    target_modules=[
        "dense",
        "o_proj",
        "qkv_proj"],
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM")

# Lora를 기본 모델에 적용
model = get_peft_model(model, lora_config)

# 참조 모델 불러오기 (필요에 따라 수정)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    return_dict=True)

# DPO 설정 정의
training_args = DPOConfig(
    beta=0.1,
    output_dir="dpo_output"
)

# Deepspeed 설정 정의 (Stage 2)
deepspeed_config = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 2,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
    }
}

# BitsandBytes Paged AdamW Optimizer 설정
optimizer_class = bnb.optim.PagedAdamW
optimizer_kwargs = {
    "lr": training_args.learning_rate
    }

dataset = get_dataset(
    dataset_name="argilla/dpo-mix-7k",
    tokenizer=tokenizer)

# DPOTrainer 초기화 (Deepspeed, PagedAdamW 적용)
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    # data_callator=,
    train_dataset=dataset["train"],  # 학습 데이터셋
    eval_dataset=dataset["test"],  # 학습 데이터셋
    tokenizer=tokenizer,
    # deepspeed=deepspeed_config,
    optimizers=(bnb.optim.PagedAdamW, {"lr": 3e-5}),
)

# DPO를 사용하여 모델 학습
dpo_trainer.train()
