import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer  # DPOTrainer 사용
from peft import LoraConfig, LoHaConfig, get_peft_model
import bitsandbytes as bnb
from data_module import get_dataset
from unsloth import FastLanguageModel


# wandb 설정
os.environ["WANDB_PROJECT"]="LLM_DPO"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="checkpoint"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="0"


# 모델과 토크나이저 불러오기
model_id = "unsloth/zephyr-sft"
model_id = "microsoft/Phi-3.5-mini-instruct"

# model = AutoModelForCausalLM.from_pretrained(model_id,
#                                              torch_dtype=torch.bfloat16,
#                                              trust_remote_code=True,
#                                              load_in_4bit=True,
#                                              return_dict=True)
# tokenizer = AutoTokenizer.from_pretrained(model_id,
#                                           trust_remote_code=True,
#                                           use_fast=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = 1024,
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# Lora 설정 정의
peft_config = LoraConfig(
    target_modules=[
        "k_proj",
        # "down_proj",
        # "gate_proj",
        "q_proj",
        "v_proj",
        # "up_proj",
        "o_proj"
    ],
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    init_lora_weights="gaussian",
    task_type="CAUSAL_LM")

# 참조 모델 불러오기 (필요에 따라 수정)
# ref_model = model

# Lora를 기본 모델에 적용
# model = get_peft_model(model, peft_config)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj",
                      "o_proj", "gate_proj", 
                      "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is currently optimized
    bias = "none",    # Bias = "none" is currently optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
)
# peft_config = LoHaConfig(
#     target_modules=[
#         "k_proj",
#         # "down_proj",
#         # "gate_proj",
#         "q_proj",
#         "v_proj",
#         # "up_proj",
#         "o_proj"
#     ],
#     r=8,
#     alpha=32,
#     task_type="CAUSAL_LM"
# )


# DPO 설정 정의
training_args = DPOConfig(
    beta=0.1,
    loss_type="ipo",
    max_prompt_length=256,
    max_target_length=1024,
    max_length=1024,
    bf16=True,
    output_dir="dpo_output",
    # deepspeed="deepspeed_config.json",
    logging_steps=50,
    num_train_epochs=50,
    gradient_accumulation_steps=64,
    generate_during_eval=True,
    dataset_num_proc=8,
    report_to="wandb",
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
    peft_config=peft_config,
    # deepspeed=deepspeed_config,
    # optimizers=(bnb.optim.PagedAdamW, {"lr": 3e-5}),
)

# DPO를 사용하여 모델 학습
dpo_trainer.train()
