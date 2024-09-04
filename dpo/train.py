import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer  # DPOTrainer 사용
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import deepspeed
from data_module import get_dataset

# 모델과 토크나이저 불러오기
model_id = "Gunulhona/Phi-Small-Merge"
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
        "k_proj",
        "down_proj",
        "gate_proj",
        "q_proj",
        "v_proj",
        "up_proj",
        "o_proj"
    ],
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM")

# 참조 모델 불러오기 (필요에 따라 수정)
# ref_model = model

# Lora를 기본 모델에 적용
# model = get_peft_model(model, lora_config)


# DPO 설정 정의
training_args = DPOConfig(
    beta=0.1,
    loss_type="ipo",
    max_length=4096,
    bf16=True,
    output_dir="dpo_output",
    deepspeed="deepspeed_config.yaml",
    logging_steps=50,
    num_train_epochs=50,
    per_device_train_batch_size=2,
)

# BitsandBytes Paged AdamW Optimizer 설정
optimizer_class = bnb.optim.PagedAdamW
optimizer_kwargs = {
    "lr": training_args.learning_rate
    }

dataset = get_dataset(
    dataset_name="Gunulhona/open_dpo_merged",
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
    peft_config=lora_config,
    # deepspeed=deepspeed_config,
    # optimizers=(bnb.optim.PagedAdamW, {"lr": 3e-5}),
)

# DPO를 사용하여 모델 학습
dpo_trainer.train()
